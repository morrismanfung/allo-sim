import numpy as np
import polars as pl
import seaborn as sns
import re
from hmmlearn import hmm, vhmm

def data_prep(asset_a_name='spy', asset_b_name='bnd'):
    df_a = pl.read_csv(f'data/{asset_a_name}.csv')
    df_a_min_date = df_a.get_column('Date').min()

    df_b = pl.read_csv(f'data/{asset_b_name}.csv')
    df_b_min_date = df_b.get_column('Date').min()

    min_possible_date = max(df_a_min_date, df_b_min_date)

    df_a_delta = df_a.with_columns(
        [
            pl.col('Adj Close').shift().alias('prev'),
            (pl.col('Adj Close') / pl.col('Adj Close').shift()).alias('delta_a')
        ]
    ).filter(
        pl.col('Date') >= min_possible_date
    ).select(
        ['Date', 'delta_a']
    )

    df_b_delta = df_b.with_columns(
        [
            pl.col('Adj Close').shift().alias('prev'),
            (pl.col('Adj Close') / pl.col('Adj Close').shift()).alias('delta_b')
        ]
    ).filter(
        pl.col('Date') >= min_possible_date
    ).select(
        ['Date', 'delta_b']
    )

    df_delta_cmb = df_a_delta.join(
        df_b_delta, on = 'Date'
    ).filter(
        pl.col('Date') > min_possible_date
    ).with_columns(
        pl.col('Date').str.to_date().alias('date'),
        pl.col('delta_a').cum_prod().alias('price_a'),
        pl.col('delta_b').cum_prod().alias('price_b'),
    ).select(
        ['date', 'delta_a', 'delta_b', 'price_a', 'price_b']
    )

    print('Earliest possible date is ', min_possible_date, '.', sep='')
    return df_delta_cmb
    
def pick_hmm(list_n_state, series_for_hmm, length, ghmm=True, gmmhmm=True):

    dict_hmm = dict()

    if ghmm == True:
        for n_state in list_n_state:
            hmm_temp = hmm.GaussianHMM(n_components=n_state, n_iter=2000, random_state=918).fit(series_for_hmm, length)
            dict_hmm[f'hmm_{n_state}'] = dict()
            dict_hmm[f'hmm_{n_state}']['score'] = hmm_temp.score(series_for_hmm)
            dict_hmm[f'hmm_{n_state}']['aic'] = hmm_temp.aic(series_for_hmm)
            dict_hmm[f'hmm_{n_state}']['bic'] = hmm_temp.bic(series_for_hmm)
        print('Gaussian Hidden Markov Models are trained and evaluated.')

    if gmmhmm == True:
        for n_state in list_n_state:
            hmm_temp = hmm.GMMHMM(n_components=n_state, n_iter=2000, random_state=918).fit(series_for_hmm, length)
            dict_hmm[f'gmmhmm_{n_state}'] = dict()
            dict_hmm[f'gmmhmm_{n_state}']['score'] = hmm_temp.score(series_for_hmm)
            dict_hmm[f'gmmhmm_{n_state}']['aic'] = hmm_temp.aic(series_for_hmm)
            dict_hmm[f'gmmhmm_{n_state}']['bic'] = hmm_temp.bic(series_for_hmm)
        print('Gaussian Mixture Hidden Markov Models are trained and evaluated.')

    return(dict_hmm)
    
def show_result(dict_hmm, list_n_state):
    pattern_hmm = re.compile('^hmm')
    exist_hmm = len([s for s in dict_hmm.keys() if pattern_hmm.match(s)]) > 0

    pattern_gmmhmm = re.compile('^gmmhmm')
    exist_gmmhmm = len([s for s in dict_hmm.keys() if pattern_gmmhmm.match(s)]) > 0

    n_states = len(dict_hmm.keys())
    
    if exist_hmm and not exist_gmmhmm:
        df_training_result = pl.DataFrame(
            {
                'model': ['GaussianHMM'] * n(list_n_state),
                'n_state': list_n_state,
                'score': [dict_hmm[f'hmm_{n_state}']['score'] for n_state in list_n_state],
                'aic': [dict_hmm[f'hmm_{n_state}']['aic'] for n_state in list_n_state],
                'bic': [dict_hmm[f'hmm_{n_state}']['bic'] for n_state in list_n_state]
            }
        )
    elif not exist_hmm and exist_gmmhmm:
        df_training_result = pl.DataFrame(
            {
                'model': ['GaussianMixtureModelHMM'] * 6,
                'n_state': list_n_state,
                'score': [dict_hmm[f'gmmhmm_{n_state}']['score'] for n_state in list_n_state],
                'aic': [dict_hmm[f'gmmhmm_{n_state}']['aic'] for n_state in list_n_state],
                'bic': [dict_hmm[f'gmmhmm_{n_state}']['bic'] for n_state in list_n_state]
            }
        )
    else:
        df_training_result = pl.DataFrame(
            {
                'model': ['GaussianHMM'] * 6 + ['GaussianMixtureModelHMM'] * 6,
                'n_state': list_n_state * 2,
                'score': [dict_hmm[f'hmm_{n_state}']['score'] for n_state in list_n_state] + [dict_hmm[f'gmmhmm_{n_state}']['score'] for n_state in list_n_state],
                'aic': [dict_hmm[f'hmm_{n_state}']['aic'] for n_state in list_n_state] + [dict_hmm[f'gmmhmm_{n_state}']['aic'] for n_state in list_n_state],
                'bic': [dict_hmm[f'hmm_{n_state}']['bic'] for n_state in list_n_state] + [dict_hmm[f'gmmhmm_{n_state}']['bic'] for n_state in list_n_state]
            }
        ).sort(
            by='bic', descending=False
        )

    return df_training_result

def simulate_performance(hmm_use, n_step=300, n_sim=1000):

    dict_sim = dict()
    rng = np.random.default_rng()
    random_states = rng.integers(0, 100_000, n_sim)

    for i in range(n_sim):
        sim_output = hmm_use.sample(n_step, random_state=random_states[i])
        sim_observation = sim_output[0]
        sim_state = sim_output[1]

        dict_sim[f'sim_{i}'] = dict()
        dict_sim[f'sim_{i}']['observation'] = sim_observation
        dict_sim[f'sim_{i}']['state'] = sim_state

    return dict_sim

def rebalance(dict_sim, n_step, n_sim, dist=(0.6, 0.4), rebalance_frequency=5, init=1):
    if len(dist) != 2:
        raise ValueError('Error: dist must be with 2 input.')
    elif dist[0] + dist[1] != 1:
        raise ValueError('Error: Elements in dist should sum up to 1.')

    for i in range(n_sim):
        amount_a = [init * dist[0]]
        amount_b = [init * dist[1]]
        amount_total = [init]
        
        for t in range(n_step):
            # If rebalance was done
            if t != 0 and t % rebalance_frequency == 0:
                rebalanced_amount_a = amount_total[-1] * dist[0]
                rebalanced_amount_b = amount_total[-1] * dist[1]
                amount_a.append(rebalanced_amount_a * dict_sim[f'sim_{i}']['observation'][t][0])
                amount_b.append(rebalanced_amount_b * dict_sim[f'sim_{i}']['observation'][t][1])
            # If rebalance was not done
            else:
                amount_a.append(amount_a[-1] * dict_sim[f'sim_{i}']['observation'][t][0])
                amount_b.append(amount_b[-1] * dict_sim[f'sim_{i}']['observation'][t][1])
        
            new_amount = amount_a[-1] + amount_b[-1]
            amount_total.append(new_amount)
    
        df_temp = pl.DataFrame({
            'amount': amount_total,
            'sim_i': [i] * (n_step + 1)
        }, strict=False).with_row_index('step')
        
        dict_sim[f'sim_{i}']['df_amount'] = df_temp

    return dict_sim
    
def plot_single_sim(sim_id, dict_sim, n_step, asset_a='A', asset_b='B'):
    price_a = [float(1)]
    price_b = [float(1)]
    
    for t in range(n_step):
        price_a.append(price_a[-1] * dict_sim[f'sim_{sim_id}']['observation'][t][0])
        price_b.append(price_b[-1] * dict_sim[f'sim_{sim_id}']['observation'][t][1])

    df_price = pl.DataFrame(
        {
            'index': list(range(len(price_a))) * 2,
            'price': price_a + price_b,
            'asset': [asset_a] * len(price_a) + [asset_b] * len(price_a)
        }
    )

    sns.lineplot(data=df_price, x='index', y='price', hue='asset')