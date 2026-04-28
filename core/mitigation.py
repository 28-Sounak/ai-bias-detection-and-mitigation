import pandas as pd

def balance_data(df, target, sensitive):
    balanced_groups = []

    for group in df[sensitive].unique():
        group_df = df[df[sensitive] == group]

        pos = group_df[group_df[target] == 1]
        neg = group_df[group_df[target] == 0]

        # Handle edge cases
        if len(pos) == 0 or len(neg) == 0:
            # Skip balancing, just keep original group
            balanced_groups.append(group_df)
            continue

        max_count = max(len(pos), len(neg))

        pos_sampled = pos.sample(max_count, replace=True)
        neg_sampled = neg.sample(max_count, replace=True)

        balanced_group = pd.concat([pos_sampled, neg_sampled])
        balanced_groups.append(balanced_group)

    return pd.concat(balanced_groups).reset_index(drop=True)