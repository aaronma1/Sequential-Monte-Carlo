
import torch

from Models import Proposal, ModulatedTarget, Tokenizer, get_toxicity_model, get_gpt2

cache_dir = "./.model_weights/"
q_hidden_size = 1024



def sis(q: Proposal, sigma, tk: Tokenizer, k_batches=10, T=20, batch_size=10, log_target=True):
    """
        perform simple importance sampling to estimate E log sigma(s_{t+1:T} | s_{0:t})
    """
    K = k_batches * batch_size
    dev = tk.device
    particles = torch.empty(K, 0, dtype=torch.int64, device=dev)
    particles, logprob_q, hidden_states = q.sample(particles, T=T, batch_size=batch_size)

    proposal_logprobs = torch.sum(logprob_q, axis=1)
    # compute target log probs
    target = torch.zeros(K, device=dev)
    # compute importance weights
    for i in range(k_batches):
        b_start = min(K, batch_size * i)
        b_end = min(K, b_start + batch_size)
        # print(b_start, b_end)
        batch = particles[b_start:b_end]
        target[b_start: b_end] = sigma.forward(
            batch, hidden_states, particle_logprobs=proposal_logprobs[b_start: b_end])

    if log_target:
        W = torch.exp(target - proposal_logprobs)
    else:
        W = torch.exp(torch.log(target) - proposal_logprobs)

    Z_SIS = torch.sum(W) / K
    print(tk.decode_gpt(q.prepend_tokens(particles)))

    return Z_SIS.item()


if __name__ == "__main__":
    gpt_model, gpt_tokenizer = get_gpt2()
    toxicity_model, toxicity_tokenizer = get_toxicity_model()
    tk = Tokenizer(q_tokenizer=gpt_tokenizer, sigma_tokenizer=toxicity_tokenizer, device=gpt_model.device)
    text = "once upon a time"
    sigma = ModulatedTarget(gpt_model, toxicity_model, tk, s0=text)
    q = Proposal(gpt_model, tk, prefix=text, temperature=1)

    # print(sis(q, sigma, tk, k_batches=1))
    K = 2
    T = 10
    dev = gpt_model.device

    print(sis(q, sigma, tk, k_batches=10, T=T, batch_size=10, log_target=True))


    # print(hidden_states.shape)

    # print(tk.decode_gpt(tokens))
    # print(tk.decode_gpt(particles))


