# does the winner change if the whole-aircraft l/d assumption is 15 or 20 instead of 18?
import study, rank, numpy as np, json
res = {}
for ld in (15.0, 20.0):
    for sc in ('gla', 'man'):
        b, base, rows = study.run(ld=ld, verbose=False, scenario=sc)
        F = [r for r in rows if r['feasible']]
        win = F[int(np.argmax(rank.score(F, rank.W_PRIORITY)))]
        res[f'{sc}_LD{ld:.0f}'] = {k: win[k] for k in ('s', 'lam', 'tw', 'dc', 'K', 'gain_cruise', 'gain_loiter', 'margin')}
        print(sc, ld, rank.fmt(win), flush=True)
json.dump(res, open('sens_ld.json', 'w'), default=float, indent=1)
