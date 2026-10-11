import torch

global_model = GPT(cfg).cuda()
outer_opt = torch.optim.SGD(
    global_model.parameters(), lr=outer_lr, momentum=0.9, nesterov=True
)
workers = [copy.deepcopy(global_model) for _ in range(k)]
inner_opts = [
    torch.optim.AdamW(w.parameters(), lr=inner_lr, weight_decay=0.1) for w in workers
]

for t in range(T):  # outer rounds
    for w, opt, loader in zip(
        workers, inner_opts, loaders
    ):  # one data shard per worker
        w.load_state_dict(global_model.state_dict())
        for _ in range(H):  # H inner steps
            x, y = loader.next()
            _, loss = w(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(w.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
    with torch.no_grad():  # pseudo-gradient = old - avg(new)
        for pg, *pws in zip(
            global_model.parameters(), *[w.parameters() for w in workers]
        ):
            pg.grad = pg.data - torch.stack([p.data for p in pws]).mean(0)
    outer_opt.step()
    outer_opt.zero_grad()
