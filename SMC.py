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

    def q_to_sigma(self,input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        sentences = self.decode_gpt(input)
        return self.tok_sigma(sentences)

    def translate_to_gpt(input: torch.Tensor) -> torch.Tensor:
        pass

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
class Proposal:
    def __init__(self, model, temperature=1):
        self.model = model
        self.temperature = temperature

    def logits(self, batch: torch.Tensor, attention_mask: torch.Tensor):
        """
            takes in a batch of sequences of shape (batch_size, max_seq_len)
            returns logits = (batch_size, px) where logits[i] represents an unnomalized density
            for p(s[i]_t | s[i]_{t-1})
        """
        with torch.no_grad():
            outputs = self.model(batch, output_hidden_states=True, attention_mask=attention_mask)
        return outputs.logits[:, -1] / self.temperature, outputs.hidden_states[-1]


class ModulatedTarget(torch.nn.Module):

    def __init__(self, model: AutoModelForCausalLM, phi: AutoModelForSequenceClassification, tk: Tokenizer):
        nn.Module.__init__(self)
        self.model = model
        self._phi = phi
        self.tk = tk

    # returns a probability distribution [p, 1-p] where p is the probability that
    # text is non-toxic
    def forward(self, particles: torch.Tensor, last_hidden_state: torch.Tensor, prefix: int, particle_logprobs=None) -> torch.Tensor:
        """
        gives sigma(s_1 ... s_T) = p(c = 1 | s_{1:T})p(s_{1:t})

        tokens: torch.Tensor of shape (batch_size, l)
        attention_mask: torch.Tensor of shape (batch_size, l)
        tokens_logprob: torch.Tensor of shape (batch_size)
        """
        x, attention_mask = self.tk.q_to_sigma(particles)
        sigma_logits = self._phi(x).logits.squeeze()
        sigma_logprob = F.logsigmoid(sigma_logits)
        # if we have logp particles, we can return immediately
        if particle_logprobs is not None:
            return sigma_logprob + particle_logprobs
        # gather logits for particles
        attention_mask = torch.ones_like(particles, device=particles.device, dtype=torch.int)
        logits = self.model(particles, attention_mask=attention_mask).logits
        # compute log q(s_1:t)
        logprob = torch.gather(logits[prefix:], 2, particles[prefix:].unsqueeze(2)).sum(axis=1)
        return sigma_logprob + logprob

    def phi(self, particles: torch.Tensor) -> torch.Tensor:
        x, attention_mask = self.tk.q_to_sigma(particles)
        sigma_logits = self._phi(x).logits.squeeze()
        return F.logsigmoid(sigma_logits)


class TwistHead(nn.Module):
    def __init__(self, phi, tk: Tokenizer, input_dim=1024, device="cpu", h=512):
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
    def forward(self, particles: torch.Tensor, last_hidden_state: torch.Tensor, prefix: int, particle_logprobs=None) -> torch.Tensor:
        # perform mean pooling on last hidden
        t = particles.shape[0] - prefix
        if t == 0:
            return torch.tensor(0.0)
        if t == self.T:
            return self.phi.phi(particles)

        mp = last_hidden_state.mean(dim=1)
        return self.mlp(mp)


def sample_t(q: Proposal, prefixes: torch.Tensor, T: int, batch_size=0):
    """
    Samples sequences si_{1:T} from q with the ith sequence having si_0 = prefix[i]

    q: Proposal distribution
    prefixes: torch.Tensor of shape (batch_size, prefix_length)
    T: sequence length of each samples
    """

    if batch_size == 0:
        batch_size = prefixes.shape[0]

    dev = prefixes.device
    K = prefixes.shape[0]
    k_batches = K // batch_size

    prefix_length = prefixes.shape[1]
    particles = torch.cat((prefixes, torch.zeros(K, T + 1, device=dev, dtype=torch.int)), dim=-1)
    logprob_q = torch.zeros(K, T + prefix_length + 1, device=dev)  # store log q(s_t | s_{0:t-1})
    hidden_states = torch.zeros(K, T + prefix_length + 1, device=dev)
    for t in range(T + 1):
        for i in range(k_batches):
            # get batch
            b_start = batch_size * i
            b_end = torch.max(K, b_start + batch_size)
            batch = particles[b_start: b_end, :t + prefix_length]
            attention_mask = torch.ones_like(batch, device=dev, dtype=torch.int)

            # predict next tokens
            next_logits, hidden = q.logits(batch.int(), attention_mask)
            particles[b_start:b_end, t + prefix_length] = torch.multinomial(
                torch.softmax(next_logits, dim=1), num_samples=1).squeeze()

            # compute and store next token log probabilities
            logprobs = F.log_softmax(next_logits, dim=1)
            logprob_q[b_start:b_end, t + prefix_length] = torch.gather(
                logprobs, 1, particles[b_start:b_end, t + prefix_length].unsqueeze(1)).squeeze()

            if t == T+1:
                hidden_states[b_start: b_end] = hidden[0]

    return particles, hidden_states, logprob_q


# should return E log 
def ctl_loss(particles: torch.Tensor, last_hidden_state: torch.Tensor, twists: TwistHead, prefix_len: int):
    t = particles.shape[1] - prefix_len

    next_particles,  = sample_t()





def train_twists(p: Proposal, sigma: ModulatedTarget, twists: TwistHead):
    pass


# SMC to sample sentences from q
# in this case p0 = base model
def smc(q: Proposal, tk: Tokenizer, s0="", k_batches=10, T=20, batch_size=10):
    dev = tk.device
    K = k_batches * batch_size
    prefixes, prefixes_attention_mask = tk.tok_q([s0 for i in range(K)])
    prefix_length = max(prefixes.shape[1], 1)

    particles = torch.cat((prefixes, torch.zeros(K, T + 1, device=dev, dtype=torch.int)), dim=-1)
    logprob_q = torch.zeros(K, T + prefix_length + 1, device=dev)  # store log q(s_t | s_{0:t-1})

    for t in range(T + 1):
        for i in range(k_batches):
            # get batch
            b_start = batch_size * i
            b_end = b_start + batch_size
            batch = particles[b_start: b_end, :t + prefix_length]
            attention_mask = torch.ones_like(batch, device=dev, dtype=torch.int)

            # predict next tokens
            next_logits, hidden_states = q.logits(batch.int(), attention_mask)
            particles[b_start:b_end, t + prefix_length] = torch.multinomial(
                torch.softmax(next_logits, dim=1), num_samples=1).squeeze()

            # compute and store next token log probabilities
            logprobs = F.log_softmax(next_logits, dim=1)
            logprob_q[b_start:b_end, t + prefix_length] = torch.gather(
                logprobs, 1, particles[b_start:b_end, t + prefix_length].unsqueeze(1)).squeeze()

        # compute importance weights
        particle_logprob_s = logprob_q[:, t + prefix_length]
        particle_logprob_s = particle_logprob_s - torch.min(particle_logprob_s)
        particle_logprob_s = torch.exp(particle_logprob_s)

        W = particle_logprob_s / torch.sum(particle_logprob_s, dim=-1)
        indices = torch.multinomial(W, num_samples=K, replacement=True)
        particles = particles[indices]

        logprob_q = logprob_q[indices]
        print(f"done iteration {t}")
    for (i, s) in enumerate(tk.decode_gpt(particles, prefix_length)):
        print(s, torch.sum(logprob_q[i]))


def sis_from_s0(q: Proposal, sigma, tk: Tokenizer, s0: str, k_batches=10, T=20, batch_size=10):
    K = k_batches * batch_size
    particles, _ = tk.tok_q([s0 for _ in range(K)])
    return sis(q, sigma, tk, particles, k_batches=k_batches, T=T, batch_size=batch_size)


def sis(q: Proposal, sigma, tk: Tokenizer, prefix: torch.Tensor, k_batches=10, T=20, batch_size=10, log_target=True):
    """
        perform simple importance sampling to estimate E log sigma(s_{t+1:T} | s_{0:t})
    """
    K = k_batches * batch_size
    prefix_length = prefix.shape[1]
    dev = prefix.device

    particles, hidden_states, logprob_q = sample_t(q, prefix, T, batch_size=batch_size)
    proposal_logprobs = torch.sum(logprob_q, axis=1)
    # compute target log probs
    target = torch.zeros(K, device=dev)
    # compute importance weights
    for i in range(k_batches):
        b_start = batch_size * i
        b_end = b_start + batch_size
        batch = particles[b_start:b_end]
        target[b_start: b_end] = sigma.forward(
            batch, hidden_states, prefix_length, particle_logprobs=proposal_logprobs[b_start: b_end])

    if log_target:
        W = torch.exp(target - proposal_logprobs)
    else:
        W = torch.exp(torch.log(target) - proposal_logprobs)

    Z_SIS = torch.sum(W) / K

    return Z_SIS.item()



import math
if __name__ == "__main__":
    gpt_model, gpt_tokenizer = get_gpt2()
    toxicity_model, toxicity_tokenizer = get_toxicity_model()
    tk = Tokenizer(q_tokenizer=gpt_tokenizer, sigma_tokenizer=toxicity_tokenizer, device=gpt_model.device)

    sigma = ModulatedTarget(gpt_model, toxicity_model, tk)
    q = Proposal(gpt_model, temperature=1)
    # smc(q, tk, s0="I", K_batches=5, batch_size=25, T=30)



    Z_sis = torch.zeros(10)
    for i in range(10):
        Z_sis[i] = sis_from_s0(q, sigma, tk, s0="dirty dirty whore", k_batches=5, T=30, batch_size=50)
        torch.cuda.memory._dump_snapshot(f"snapshot{i}.pickle")
        torch.cuda.empty_cache()
        print(Z_sis[i])

    print(Z_sis, torch.mean(Z_sis), torch.std(Z_sis))


