import math

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

cache_dir = "./.model_weights/"


def get_gpt2():
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium', cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2-medium', cache_dir=cache_dir)
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def get_toxicity_model():
    tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel", cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/ToxicityModel", cache_dir=cache_dir)
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


class Tokenizer():
    def tok_q(self, inputs: list[str]) -> (torch.Tensor, torch.Tensor):
        tokens = self.q_tokenizer(
            inputs,
            padding="longest",
            return_tensors="pt",
            return_attention_mask="true",
        )
        tokens.to(self.device)
        return tokens.input_ids, tokens.attention_mask

    def tok_sigma(self, input: list[str]) -> (torch.Tensor, torch.Tensor):
        tokens = self.sigma_tokenizer(
            input,
            truncation=True,
            max_length=512,
            padding="longest",
            return_token_type_ids=False,
            return_tensors="pt",
            return_attention_mask=True)
        tokens.to(self.device)
        return tokens.input_ids, tokens.attention_mask

    def q_to_sigma(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        sentences = self.decode_gpt(input)
        return self.tok_sigma(sentences)

    def decode_gpt(self, input_ids: torch.Tensor) -> list[str]:
        return [self.q_tokenizer.decode(input_ids[i, :], skip_special_tokens=False)
                for i in range(input_ids.shape[0])]

    def __init__(self, q_tokenizer, sigma_tokenizer, device="cpu"):
        self.sigma_tokenizer = sigma_tokenizer
        self.q_tokenizer = q_tokenizer
        self.device = device


# represents a probability distribution q(s_{1}...s_{T}_
# gives logits for a marginal q(s_{t+1} | s_1 ... s_{t})
# we can compute logits for proposal distribution
# gpt 2 in this case
class Proposal(nn.Module):
    def __init__(self, model, tk: Tokenizer, temperature=1, prefix=""):
        nn.Module.__init__(self)
        self.model = model
        self.temperature = temperature
        self.tk = tk
        self.prefix, _ = tk.tok_q([prefix])
        self.prefix = self.prefix.squeeze()

    def forward(self, tokens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        tokens = self.prepend_tokens(tokens)
        attention_mask = torch.ones_like(tokens, device=tokens.device, dtype=torch.int)

        with torch.no_grad():
            outputs = self.model(tokens, output_hidden_states=True, attention_mask=attention_mask, use_cache=True)
        return outputs.logits[:, -1], outputs.hidden_states[-1]

    def prepend_tokens(self, tokens):
        k = tokens.shape[0]
        k_prefixes = self.prefix.unsqueeze(0).repeat(k, 1)
        tokens = torch.cat((k_prefixes, tokens), dim=1)
        return tokens

    def sample(self, particles, batch_size=0, T=1):
        if batch_size == 0:
            batch_size = particles.shape[0]
        dev = particles.device
        K = particles.shape[0]
        k_batches = math.ceil(K / batch_size)
        logprobs = torch.zeros(K, T, device=dev)
        pl = particles.shape[1]
        particles = torch.cat((particles, torch.zeros(K, T, dtype=torch.int64, device=dev)), dim=-1)
        for i in range(k_batches):
            # get batch
            b_start = min(K, batch_size * i)
            b_end = min(K, b_start + batch_size)

            batch_hidden = None
            for t in range(T):
                # print(particles)
                batch = particles[b_start: b_end, :t + pl]

                next_logits, batch_hidden = self.forward(batch)

                particles[b_start:b_end, t + pl] = torch.multinomial(
                    torch.softmax(next_logits, dim=1), num_samples=1).squeeze()
                particles_dist = F.log_softmax(next_logits, dim=1)
                logprobs[b_start:b_end, t + pl] = torch.gather(
                    particles_dist, dim=1, index=particles[b_start:b_end, t + pl].unsqueeze(1)
                ).squeeze()

            if i == 0:
                hidden_states = batch_hidden
            else:
                hidden_states = torch.cat((hidden_states, batch_hidden), dim=0)
        # print(hidden_states.shape)

        return particles, logprobs, hidden_states


class ModulatedTarget(nn.Module):

    def __init__(self, model: AutoModelForCausalLM, phi: AutoModelForSequenceClassification, tk: Tokenizer, s0=""):
        nn.Module.__init__(self)
        self.model = model
        self._phi = phi
        self.tk = tk
        self.prefix = tk.tok_q([s0])[0].squeeze().int()

    def prepend_tokens(self, tokens):
        k = tokens.shape[0]
        k_prefixes = self.prefix.unsqueeze(0).repeat(k, 1)
        tokens = torch.cat((k_prefixes, tokens), dim=1)
        return tokens

    # returns a probability distribution [p, 1-p] where p is the probability that
    # text is non-toxic
    def forward(
            self, particles: torch.Tensor, last_hidden_state: torch.Tensor, particle_logprobs=None
    ) -> torch.Tensor:
        """
        gives sigma(s_1 ... s_T) = p(c = 1 | s_{1:T})p(s_{1:t})

        tokens: torch.Tensor of shape (batch_size, l)
        attention_mask: torch.Tensor of shape (batch_size, l)
        tokens_logprob: torch.Tensor of shape (batch_size)
        """

        particles = self.prepend_tokens(particles)
        x, attention_mask = self.tk.q_to_sigma(particles)
        sigma_logits = self._phi(x, attention_mask=attention_mask).logits.squeeze()
        sigma_logprob = F.logsigmoid(sigma_logits)
        # if we have logp particles, we can return immediately
        if particle_logprobs is not None:
            return sigma_logprob + particle_logprobs

        # gather logits for particles
        attention_mask = torch.ones_like(particles, device=particles.device, dtype=torch.int)
        logits = self.model(particles, attention_mask=attention_mask).logits
        # compute log q(s_1:t)
        prefix = self.prefix.shape[0]
        logprob = torch.gather(logits[prefix:], 2, particles[prefix:].unsqueeze(2)).sum(axis=1)
        return sigma_logprob + logprob

    def phi(self, particles: torch.Tensor) -> torch.Tensor:
        particles = self.prepend_tokens(particles)
        x, attention_mask = self.tk.q_to_sigma(particles)
        sigma_logits = self._phi(x).logits.squeeze()
        return F.logsigmoid(sigma_logits)


class TwistHead(nn.Module):
    def __init__(self, phi, tk: Tokenizer, input_dim=1024, device="cpu", h=512):
        nn.Module.__init__(self)
        self.device = device
        self.mlp = nn.Parameter(
            nn.Sequential(
                nn.Linear(input_dim, h),
                nn.Relu(),
                nn.Linear(h, h),
                nn.Relu(),
                nn.Linear(h, h),
                nn.Relu(),
                nn.Linear(h, 1)
            )
        ).to(device)

    # return log psi (s_{1:t})
    def forward(
            self,
            particles: torch.Tensor,
            last_hidden_state: torch.Tensor,
            prefix_len: int,
            particle_logprobs=None
    ) -> torch.Tensor:

        # perform mean pooling on last hidden
        t = particles.shape[0] - prefix_len
        if t == 0:
            return torch.tensor(0.0)
        if t == self.T:
            return self.phi.phi(particles)

        mp = last_hidden_state.mean(dim=1)
        return self.mlp(mp)
