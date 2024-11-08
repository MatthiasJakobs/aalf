Ls = {
    'weather': 9,
    'nn5_daily': 9,
    #'tourism_yearly': 2,
    'tourism_quarterly': 5,
    'tourism_monthly': 15,
    'dominick': 10,
}

tsx_to_gluon = {
    'weather': 'weather',
    'nn5_daily': 'nn5_daily_without_missing',
    'tourism_yearly': 'tourism_yearly',
    'tourism_quarterly': 'tourism_quarterly',
    'tourism_monthly': 'tourism_monthly',
    'dominick': 'dominick'
}

all_datasets = list(Ls.keys())
dev_datasets = ['weather', 'nn5_daily', 'tourism_monthly']