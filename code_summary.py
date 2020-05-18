import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import statsmodels.formula.api as sm
import ffn
import warnings
warnings.filterwarnings('ignore')
import os
os.getcwd()
os.chdir("/Users/heony/Desktop/final")

##################### Data ##################### 
df = pd.read_excel("etf_list.xlsx")
ticker_c = [i for i in df.country.dropna()]
ticker_i = [i for i in df.industry.dropna()]
ticker_b = ['SHY']

price_c = ffn.data.get(ticker_c, start='2005-12-01', end='2018-12-31', common_dates=False)
price_i = ffn.data.get(ticker_i, start='2005-12-01', end='2018-12-31', common_dates=False)
price_b = ffn.data.get(ticker_b, start='2005-12-01', end='2018-12-31', common_dates=False)

##################### Pre-step ##################### 
def get_RMPF_weight(price, N, pct1, pct2):
    
    '''
    make equal weight portfolio using relative momentum score
    '''
    
    monPrice = price.resample('M').last()
    RMscore = ((monPrice-monPrice.shift(N))/monPrice.shift(N)).shift(1).iloc[N+1:]
    
    weight = pd.DataFrame(columns = monPrice.columns)
    for i in RMscore.index:
        target = RMscore.loc[i].sort_values(ascending=False).dropna()
        tg_len = len(target)
        ticker = target[int(pct1*tg_len):int(pct2*tg_len)].index
        weight_i = pd.DataFrame(1/ticker.size, index = [i], columns = ticker)
        weight = pd.concat([weight, weight_i])
        
    return weight

def get_group_PF_return(price, N):
    
    '''
    get P1, P2, P3 return
    '''
    w1 = get_RMPF_weight(price, N, 0, 0.3)
    w2 = get_RMPF_weight(price, N, 0.3, 0.7)
    w3 = get_RMPF_weight(price, N, 0.7, 1)
    
    m_ret = price.resample('M').last().pct_change().reindex(w1.index)
    
    pf1 = (w1*m_ret).sum(axis=1)
    pf2 = (w2*m_ret).sum(axis=1)
    pf3 = (w3*m_ret).sum(axis=1)
    pf = pd.concat([pf1,pf2,pf3],axis=1)
    pf.columns=['P1','P2','P3']
    
    return pf

def get_results(price, N, gp):
    
    '''
    produce results 
    '''
    
    ret = get_group_PF_return(price, N)
    ret = ret.reindex(price.resample('M').last().loc['2006-12-29':].index).fillna(0)
    cum_ret = ret.add(1).cumprod()
    cum_ret.plot(grid=True, figsize=(12,6), title = '{0}, K={1}'.format(gp, N))
    plt.savefig('{0}_{1}'.format(gp, N))
    ffn.core.calc_stats(cum_ret).to_csv(sep=',', path ='{0}_{1}.csv'.format(gp, N))

    return True

#get_results(price_c, 12, 'Country')

##################### Strategy ##################### 
def relativeMomentum(price, N):
    
    '''
    this function calculate average relative momentum
    '''
    
    monPrice = price.resample('M').last()
    relativeMom = pd.DataFrame(0,index = monPrice.index, columns = monPrice.columns)
    for i in range(1,N+1):
        mom_i = (monPrice-monPrice.shift(i))/monPrice.shift(i)
        relativeMom = relativeMom + mom_i
    
    return (relativeMom/N).iloc[N:]

def absoluteMomentum(price, N):
    
    '''
    this function calculate average absolute momentum
    '''
    
    monPrice = price.resample('M').last()
    absoluteMom = pd.DataFrame(0,index = monPrice.index, columns = monPrice.columns)
    for i in range(1, N+1):
        absoluteMom = absoluteMom + np.where(monPrice / monPrice.shift(i) >1, 1, 0)
    
    return (absoluteMom/N).iloc[N:]

def topPercentage(series, pct=0.3):
    
    '''
    get ticker we want (highly ranked)
    '''
    
    obj = series.sort_values(ascending=False).dropna()
    
    return obj[:int(len(obj)*pct)]

def groupPfWeight(price, N, pct=0.3, trend=0.5, fixed_weight=0.3):
    
    '''
    step 1 & 2
    '''
    
    rm = relativeMomentum(price,N)
    am = absoluteMomentum(price,N)
    
    weight = pd.DataFrame(index = rm.index,columns = rm.columns)
    for i in rm.index:
        ind_1 = topPercentage(rm.loc[i],pct).index
        ind_2 = am[am>trend].loc[i].index
        ind_w = ind_1.intersection(ind_2)
        index_for_bond = ind_1.difference(ind_2)
        weight.loc[i,ind_w] = (am.loc[i,ind_w])
        weight.loc[i,'bond'] = am.loc[i,index_for_bond].sum() + ind_w.size - am.loc[i,ind_w].sum()
    weight = weight.div(weight.sum(axis=1),axis=0)
    weight = weight.shift(1).iloc[1:].fillna(0)#해당월의 웨이트가 그달의 웨이트
    weight *= (1-fixed_weight)
    weight.bond += fixed_weight
    
    return weight

def Calculate_Bond_return(BondPrice, freq='M'):
    
    '''
    this function caculate equally weighted return of bond universe
    '''
    
    bond_ret = BondPrice.resample(freq).last().pct_change()
    
    bond_total_ret = pd.DataFrame(index = bond_ret.index, columns=['bond'])
    for i in range(bond_ret.index.size):
        if bond_ret.iloc[i].dropna().index.size == 0:
            bond_total_ret.iloc[i] = 0
        else:
            bond_total_ret.iloc[i] = bond_ret.iloc[i].sum()/(bond_ret.iloc[i].dropna().size)
        
    return bond_total_ret

def generate_month_list(timeIndex, M=6):
    
    monthList = []
    for i in range(M+1):
        monthList.append(timeIndex.to_pydatetime() - relativedelta(months=+i))
    
    return monthList[::-1]

def recalculate_weight(price, BondPrice, N, M=6):
    
    '''
    step 3
    '''
    
    gp_weight = groupPfWeight(price, N)
    gp_mret = price.resample('M').last().pct_change()
    bond_mret = Calculate_Bond_return(BondPrice) 
    tot_mret = pd.concat([gp_mret,bond_mret],axis=1)

    for t in gp_weight.index:
        t_list = generate_month_list(t, M)
        weight_t = pd.DataFrame(gp_weight.loc[t]).T.reindex(tot_mret.loc[t_list].index).bfill()
        pf_t = (weight_t * tot_mret.loc[t_list]).sum(axis=1).add(1).cumprod()
        
        score = 0
        for m in range(1,M+1):
            score += np.where(pf_t/pf_t.shift(m) > 1, 1, 0)[-1]
        score /= M
        gp_weight.loc[t] *= score
        gp_weight.loc[t,'bond'] += (1-score)
    
    return gp_weight

def Backtesting(price1, price2, BondPrice, N=12, M=6, init_balance = 1000000):
    
    '''
    Backtest strategy - calculating daily return
    '''
    
    weight1 = recalculate_weight(price1, BondPrice, N, M)
    weight2 = recalculate_weight(price2, BondPrice, N, M)
    weight = pd.concat([weight1.iloc[:,:weight1.columns.size-1],weight2.iloc[:,:weight2.columns.size-1],weight1.bond+weight2.bond],axis=1)/2
    start_day = generate_month_list(weight.index[0], 1)[0]
    Price = pd.concat([price1, price2, BondPrice],axis=1)
    mPrice = Price.resample('M').last().loc[start_day:]
    Bond_mPrice = mPrice.loc[:,BondPrice.columns]
    
    b_w = pd.DataFrame(columns = Bond_mPrice.columns)
    for i in range(weight.index.size):
        b_w_i = pd.DataFrame(weight.bond.iloc[i]/Bond_mPrice.iloc[i].dropna().index.size, index=[weight.index[i]], columns = Bond_mPrice.iloc[i].dropna().index)
        b_w = pd.concat([b_w,b_w_i],axis=0)
        
    weight_new = pd.concat([weight.drop(['bond'], axis=1), b_w],axis=1)
    
    balance = pd.DataFrame(index = Bond_mPrice.index, columns = ['balance'])
    balance.iloc[0] = init_balance
    holding_num = pd.DataFrame(columns = mPrice.columns)
    
    for i in range(balance.index.size-1):
        holding_num_i = (weight_new.iloc[i]*balance.iloc[i,0]).div(mPrice.iloc[i])
        holding_num = pd.concat([holding_num,pd.DataFrame(holding_num_i, columns=[balance.index[i+1]]).T],axis=0)
        balance.iloc[i+1] = (mPrice.iloc[i+1]*holding_num_i).sum()
    
    target_price = Price.loc[start_day:]
    expanded_hn = holding_num.fillna(0).reindex(mPrice.index).resample('D').last().bfill().reindex(target_price.index)
    daily_balance = (expanded_hn*target_price).sum(axis=1)
    init_bal = pd.Series(init_balance, index = [mPrice.index[0]])
    daily_balance = pd.concat([init_bal, daily_balance])
    
    return pd.DataFrame(daily_balance.pct_change().fillna(0), columns=['return'])

def Calculate_equal_weight(price, fw):
    
    '''
    calculate equal weight for benchmark
    '''
    
    mPrice = price.resample('M').last()
    
    weight = pd.DataFrame(columns = mPrice.columns)
    for d in mPrice.index:
        tg = mPrice.loc[d].dropna()
        w = pd.DataFrame(fw/tg.size, index = [d], columns = tg.index)
        weight = pd.concat([weight, w], sort=False)

    return weight.shift(1).iloc[1:]

def Calculate_BM_return(price1, price2, BondPrice, fixed_weight = 0.3, init_balance = 1000000):
    
    '''
    calculating benchmark return
    '''
    
    weight1 = Calculate_equal_weight(price1, (1-fixed_weight)/2)  
    weight2 = Calculate_equal_weight(price2, (1-fixed_weight)/2)
    weightb = pd.DataFrame(fixed_weight, index = weight1.index, columns = BondPrice.columns)
    
    weight = pd.concat([weight1,weight2,weightb] ,axis=1)
    start_day = generate_month_list(weight.index[0], 1)[0]
    Price = pd.concat([price1, price2, BondPrice],axis=1)
    mPrice = Price.resample('M').last().loc[start_day:]
        
    balance = pd.DataFrame(index = mPrice.index, columns = ['balance'])
    balance.iloc[0] = init_balance
    holding_num = pd.DataFrame(columns = mPrice.columns)
    
    for i in range(balance.index.size-1):
        holding_num_i = (weight.iloc[i]*balance.iloc[i,0]).div(mPrice.iloc[i])
        holding_num = pd.concat([holding_num, pd.DataFrame(holding_num_i, columns = [balance.index[i+1]]).T], axis=0)
        balance.iloc[i+1] = (mPrice.iloc[i+1]*holding_num_i).sum()
        
    target_price = Price.loc[start_day:]
    expanded_hn = holding_num.fillna(0).reindex(mPrice.index).resample('D').last().bfill().reindex(target_price.index)
    daily_balance = (expanded_hn*target_price).sum(axis=1)

    return pd.DataFrame(daily_balance.pct_change().fillna(0), columns=['return'])

pf = Backtesting(price_c, price_i, price_b, 12, 6)
bm = Calculate_BM_return(price_c, price_i, price_b).reindex(pf.index).fillna(0)

res = pd.concat([pf,bm],axis=1)
res.columns = ['PF(12,6)','Benchmark']
res = res.iloc[1:].reindex(price_c.loc['2006-12-29':].index).fillna(0)

##################### analysis #####################
cum_ret = res.add(1).cumprod()
cum_ret.plot(grid=True, figsize=(12,6), title='Cumulative Return')
plt.savefig('Strategy')    
ffn.core.calc_stats(cum_ret).to_csv(sep=',', path ='strategy_summary.csv')
 
def regression():
    
    '''
    regress the strategy with benchmark, size and value factor
    '''
    
    res_mret = res.add(1).resample('M').prod()-1
    
    FF3data = pd.read_csv("F-F_Research_Data_Factors.csv", header = 2, index_col = 0).loc[:'201902']
    FF3data.index = pd.to_datetime(FF3data.index, format='%Y%m')
    FF3data = (FF3data.astype(float)/100).resample('M').last()
    FF3data = FF3data.reindex(res_mret.index)

    data = pd.concat([res_mret.sub(FF3data.RF,axis=0),FF3data[['SMB','HML']]],axis=1).iloc[1:]
    data.columns=['PFX','BMX','SMB','HML']
    
    CAPM = sm.ols(formula = 'PFX ~ BMX', data = data).fit()
    CAPMres = pd.concat([CAPM.params,CAPM.tvalues,CAPM.pvalues],axis=1)
    CAPMres.columns = ['coef', 't stat','p value']
    
    FF3 = sm.ols(formula = 'PFX ~ BMX + SMB + HML', data = data).fit()
    FF3res = pd.concat([FF3.params,FF3.tvalues,FF3.pvalues],axis=1)
    FF3res.columns = ['coef', 't stat','p value']
    
    print( 'CAPM results : \n', np.round(CAPMres, 5) )
    print( 'FamaFrench 3 Factor results : \n', np.round(FF3res, 5) )
    
    return True

regression()

