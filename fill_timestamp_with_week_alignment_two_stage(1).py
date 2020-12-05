# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 01:51:10 2020

@author: MI
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
import itertools

def plot_speed_curve_of_every_day(speed_df):
    def plot_speed_curve(x, y, title, save_file):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = make_interp_spline(x, y)(x_smooth)
        ax.plot(x_smooth, y_smooth, 'r')
        ax.scatter(x, y)
        ax.grid(True)
        ax.set_title(title)
        fig.savefig(save_file, dpi=300)
        plt.close()
    
    save_path = Path('./speed_curve_of_every_day')
    save_path.mkdir(exist_ok=True)
    speed_stats = {}
    for _, (_, date, speed) in speed_df.iterrows():
        day, time = date.split(' ')
        time = int(time.split(':')[0])
        speed_stats[day] = {**speed_stats.get(day, {}), time:speed}
        
    for day, speed_dict in tqdm(speed_stats.items()):
        week = datetime.datetime.strptime(day, "%d/%m/%Y").strftime("%w")
        x, y = map(np.array, zip(*[[x, y] for x, y in speed_dict.items()]))
        save_file = save_path / ('_'.join(day.split('/')[::-1]) + '.png')
        plot_speed_curve(x, y, f'{day} week{week}', save_file)

def fill_miss_speed(df_x, df_y):
    fill_df_y = df_y.loc[df_y.speed==-1].copy()
    if not fill_df_y.time.isin(df_x.time).all():
        return np.inf, df_y
    ori_df_y = df_y.loc[df_y.speed!=-1].copy()
    speed_x = df_x.speed[df_x.time.isin(ori_df_y.time)].to_numpy()
    speed_y = ori_df_y.speed[ori_df_y.time.isin(df_x.time)].to_numpy()
    norm_x = (speed_x - speed_x.min()) / (speed_x.max() - speed_x.min())
    norm_y = (speed_y - speed_y.min()) / (speed_y.max() - speed_y.min())
    mse = ((norm_x - norm_y) ** 2).mean() * (speed_y.max() - speed_y.min())**2
    new_speed = []
    for time, speed in df_y[['time', 'speed']].values:
        if speed != -1:
            new_speed.append(speed)
        else:
            speed_ = df_x.loc[df_x.time==time, 'speed'].values[0]
            speed_ = (speed_ - speed_x.min()) / (speed_x.max() - speed_x.min())
            speed_ = speed_ * (speed_y.max() - speed_y.min()) + speed_y.min()
            new_speed.append(speed_)
            
    new_y = df_y.copy()
    new_y['speed'] = new_speed
    return mse, new_y

def date_split(train_df):
    holiday = [[2017, 1, 2], [2017, 1, 28], [2017, 1, 30], [2017, 1, 31], [2017, 4, 4], [2017, 4, 14],\
               [2017, 4, 15], [2017, 4, 17], [2017, 5, 1], [2017, 5, 3], [2017, 5, 30], [2017, 7, 1],\
               [2017, 10, 2], [2017, 10, 5], [2017, 10, 28], [2017, 12, 25], [2017, 12, 26],\
               [2018, 1, 1], [2018, 2, 16], [2018, 2, 17], [2018, 2, 19], [2018, 3, 30], [2018, 3, 31],\
               [2018, 4, 2], [2018, 4, 5], [2018, 5, 1], [2018, 5, 22], [2018, 6, 18], [2018, 7, 2],\
               [2018, 9, 25], [2018, 10, 1], [2018, 10, 17], [2018, 12, 25], [2018, 12, 26]]
    
    train_df['time'] = train_df.date.apply(lambda x:int(x.split(' ')[1].split(':')[0]))
    train_df['day'] = train_df.date.apply(lambda x:int(x.split(' ')[0].split('/')[0]))
    train_df['month'] = train_df.date.apply(lambda x:int(x.split(' ')[0].split('/')[1]))
    train_df['year'] = train_df.date.apply(lambda x:int(x.split(' ')[0].split('/')[2]))
    train_df['week'] = train_df.date.apply(lambda x:int(datetime.datetime.strptime(x, "%d/%m/%Y %H:%M").strftime("%w")))
    train_df['day_month'] = train_df.date.apply(lambda x:'_'.join(x.split('/')[:2]))
    train_df['day_month_year'] = train_df.date.apply(lambda x:x.split(' ')[0])
    train_df['holiday'] = train_df.apply(lambda x: int([x.year, x.month, x.day] in holiday), axis=1)
    
def plot_fill_res(fill_res, save_path):
    def get_save_name(df):
        return '_'.join(df.date.iloc[0].split(" ")[0].split("/")[::-1]) + f'_week{df.week.iloc[0]}'
    
    col = len(fill_res)
    fig = plt.figure(figsize=(24, 8))
    save_path.mkdir(exist_ok=True)
    save_name = get_save_name(fill_res[0][1])
    
    for ind, (mse, fill_df, template_df) in enumerate(fill_res, 1):
        fill_df = fill_df.sort_values(by='time')
        template_df = template_df.sort_values(by='time')
        ax = fig.add_subplot(2, col, ind)
        x = fill_df.time.loc[fill_df.flag=='train'].to_numpy()
        y = fill_df.speed.loc[fill_df.flag=='train'].to_numpy()
        ax.scatter(x, y)
        x = fill_df.time.loc[fill_df.flag=='test'].to_numpy()
        y = fill_df.speed.loc[fill_df.flag=='test'].to_numpy()
        ax.scatter(x, y, color="w", linewidths=1, edgecolors='orange')
        x_smooth = np.linspace(fill_df.time.min(), fill_df.time.max(), 100)
        y_smooth = make_interp_spline(fill_df.time.to_numpy(), fill_df.speed.to_numpy())(x_smooth)
        ax.plot(x_smooth, y_smooth, 'r')
        ax.grid(True)
        ax.set_title(get_save_name(fill_df) + f'_mse: {mse:.4f}')
        
        ax = fig.add_subplot(2, col, ind+col)
        x = template_df.time.to_numpy()
        y = template_df.speed.to_numpy()
        ax.scatter(x, y)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = make_interp_spline(x, y)(x_smooth)
        ax.plot(x_smooth, y_smooth, 'r')
        ax.grid(True)
        ax.set_title(get_save_name(template_df) + f'_mse: {mse:.4f}')
    fig.savefig(save_path / save_name, dpi=300)
    plt.close()

def get_mse_weight(*mse):
    weight = np.array([1/mse_ for mse_ in mse])
    weight /= weight.sum()
    return weight

if __name__ == '__main__':
    train_csv_path = 'E:/HKUST/5001/kaggle/train.csv'
    test_csv_path = 'E:/HKUST/5001/kaggle/test.csv'
    submission_path = 'E:/HKUST/5001/kaggle/submissionn.csv'
       
    train_csv = pd.read_csv(train_csv_path)
    if not Path('./speed_curve_of_every_day').exists():
        plot_speed_curve_of_every_day(train_csv)
    date_split(train_csv)
    train_2017_df = train_csv.loc[train_csv.year==2017].copy()
    train_2018_df = train_csv.loc[train_csv.year==2018].copy()
    train_2018_df['flag'] = 'train'
    test_csv = pd.read_csv(test_csv_path)
    date_split(test_csv)
    test_csv['speed'] = -1
    test_csv['flag'] = 'test'
    train_2018_df = pd.concat([test_csv, train_2018_df], axis=0, ignore_index=True)
    
    new_2018_df = []
    for day_month, day_2018_df in tqdm(train_2018_df.groupby('day_month')):
        week = [day_2018_df.week.iloc[0]]
        month = day_2018_df.month.iloc[0]
        month_trans = lambda x:(x - 1) % 12 +1
        month = [month_trans(month - 1), month, month_trans(month + 1)]
        if day_2018_df.holiday.iloc[0]:
            align_2017_df = train_2017_df.loc[train_2017_df.holiday==1].copy()
        else:
            align_2017_df = train_2017_df.loc[(train_2017_df.week.isin(week) & train_2017_df.month.isin(month)) & (train_2017_df.holiday==0)].copy()
        fill_res = []
        for _, day_2017_df in align_2017_df.groupby('day_month'):
            fill_res.append(fill_miss_speed(day_2017_df, day_2018_df) + (day_2017_df,))
        fill_res = sorted(fill_res, key=lambda x:x[0])
        plot_fill_res(fill_res[:5], Path('./fill_result_2018'))
        best_fill_2018_df = fill_res[0][1].drop(columns='speed')
        # weight = get_mse_weight(fill_res[0][0], fill_res[1][0], fill_res[2][0])
        # best_fill_2018_df['speed'] = weight[0] * fill_res[0][1].speed + weight[1] * fill_res[1][1].speed + weight[2] * fill_res[2][1].speed
        best_fill_2018_df['speed'] = (fill_res[0][1].speed + fill_res[1][1].speed + fill_res[2][1].speed) / 3
        new_2018_df.append(best_fill_2018_df)
    new_2018_df = pd.concat(new_2018_df, axis=0, ignore_index=True)
    
    
    missing_2017_df = pd.DataFrame([[0, f"{d_m.replace('_', '/')}/2017 {t}:00", -1, 'test'] for t, d_m in itertools.product(range(24), train_2017_df.day_month.unique())\
                                        if not f"{d_m.replace('_', '/')}/2017 {t}:00" in train_2017_df.date.tolist()], columns=['id', 'date', 'speed', 'flag'])
    date_split(missing_2017_df)
    train_2017_df['flag'] = 'train'
    all_2017_df = pd.concat([train_2017_df, missing_2017_df], axis=0, ignore_index=True).sort_values(by=['month', 'day', 'time'])
    all_2017_df['id'] = range(all_2017_df.shape[0])
    new_2017_df = []
    for day_month, day_2017_df in tqdm(all_2017_df.groupby('day_month')):
        if not day_2017_df.speed.isin([-1]).any():
            new_2017_df.append(day_2017_df)
            continue
        week = [day_2017_df.week.iloc[0]]
        month = day_2017_df.month.iloc[0]
        month_trans = lambda x:(x - 1) % 12 +1
        month = [month_trans(month - 1), month, month_trans(month + 1)]
        # the weeks of 2017 missing timestamps are different between each other
        if day_2017_df.holiday.iloc[0]:
            align_df = pd.concat([train_2017_df.loc[(train_2017_df.holiday==1) & (train_2017_df.day_month != day_month)].copy(),\
                                  new_2018_df.loc[new_2018_df.holiday==1].copy()], axis=0, ignore_index=True)
        else:
            align_df = pd.concat([train_2017_df.loc[train_2017_df.week.isin(week) & train_2017_df.month.isin(month) & (train_2017_df.holiday==0) & (train_2017_df.day_month != day_month)].copy(),\
                                  new_2018_df.loc[(new_2018_df.week.isin(week) & new_2018_df.month.isin(month)) & (new_2018_df.holiday==0)].copy()], axis=0, ignore_index=True)
        fill_res = []
        for _, align_df_ in align_df.groupby('day_month_year'):
            fill_res.append(fill_miss_speed(align_df_, day_2017_df) + (align_df_,))
        fill_res = sorted(fill_res, key=lambda x:x[0])
        plot_fill_res(fill_res[:5], Path('./fill_result_2017'))
        best_fill_2017_df = fill_res[0][1].drop(columns='speed')
        best_fill_2017_df['speed'] = (fill_res[0][1].speed + fill_res[1][1].speed + fill_res[2][1].speed) / 3
        new_2017_df.append(best_fill_2017_df)
    new_2017_df = pd.concat(new_2017_df, axis=0, ignore_index=True)
    
    new_2018_it_df = []
    for day_month, day_2018_df in tqdm(train_2018_df.groupby('day_month')):
        week = [day_2018_df.week.iloc[0]]
        month = day_2018_df.month.iloc[0]
        month_trans = lambda x:(x - 1) % 12 +1
        month = [month_trans(month - 1), month, month_trans(month + 1)]
        if day_2018_df.holiday.iloc[0]:
            align_df = pd.concat([new_2018_df.loc[(new_2018_df.holiday==1) & (new_2018_df.day_month != day_month)].copy(),\
                                  new_2017_df.loc[new_2017_df.holiday==1].copy()], axis=0, ignore_index=True)
        else:
            align_df = pd.concat([new_2018_df.loc[(new_2018_df.week.isin(week) & new_2018_df.month.isin(month)) & (new_2018_df.holiday==0)& (new_2018_df.day_month != day_month)].copy(),\
                                  new_2017_df.loc[(new_2017_df.week.isin(week) & new_2017_df.month.isin(month)) & (new_2017_df.holiday==0)].copy()], axis=0, ignore_index=True)
        fill_res = []
        for _, align_df_ in align_df.groupby('day_month_year'):
            fill_res.append(fill_miss_speed(align_df_, day_2018_df) + (align_df_,))
        fill_res = sorted(fill_res, key=lambda x:x[0])
        plot_fill_res(fill_res[:5], Path('./fill_result_2018_it'))
        best_fill_2018_df = fill_res[0][1].drop(columns='speed')
        best_fill_2018_df['speed'] = (fill_res[0][1].speed + fill_res[1][1].speed + fill_res[2][1].speed) / 3
        new_2018_it_df.append(best_fill_2018_df)
    new_2018_it_df = pd.concat(new_2018_it_df, axis=0, ignore_index=True)
    
    new_2017_df.to_csv('./filled_2017.csv', index=False)
    new_2018_df.to_csv('./filled_2018.csv', index=False)
    new_2018_it_df.to_csv('./filled_2018_it.csv', index=False)
    
    submission_df = new_2018_it_df.loc[new_2018_it_df.flag=='test', ['id', 'speed']].sort_values(by='id')
    submission_df.to_csv(submission_path, index=False)
