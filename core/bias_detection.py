def demographic_parity(df, target, sensitive):
    groups = df[sensitive].unique()
    rates = {}

    for g in groups:
        subset = df[df[sensitive] == g]

        if len(subset) == 0:
            rates[g] = 0
        else:
            rates[g] = subset[target].mean()
        
    return rates

def disparate_impact(rates):
    values = list(rates.values())

    if len(values) < 2:
        return 1.0

    min_rate = min(values)
    max_rate = max(values)

    if max_rate == 0:
        return 0
    
    return min_rate / max_rate