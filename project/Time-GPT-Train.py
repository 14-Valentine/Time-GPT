import pandas as pd
import numpy as np
from datetime import timedelta
from nixtla import NixtlaClient
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="DataFrameGroupBy.apply operated on the grouping columns.*"
)

# ========== ตั้งค่า TimeGPT ==========
nixtla_client = NixtlaClient(
    api_key="nixak-Mk36tuxBDldx106CdgUQRQqzzUDx2OHKChVseoPdlKjDi0rL7sXD5GOCwacjOJVNA3uzJUHOADTQFyQV"   # <-- แก้เป็น key ของคุณ
)

# ========== STEP 1: อ่านไฟล์ Kansai ==========
df_ks = pd.read_csv(
    r"D:\Time-GPT\dataset\IM_DA_KS_ALL.csv",
    parse_dates=['DATETIME']      # ใช้ DATETIME เป็นเวลาแบบเต็ม
)

df_ks = df_ks.rename(columns={
    'DATETIME': 'ds',   # เวลา
    'IM_KS': 'load'     # ค่าไฟ Kansai
})
df_ks['city'] = 'Kansai'

# ========== STEP 1b: อ่านไฟล์ Tokyo ==========
df_tk = pd.read_csv(
    r"D:\Time-GPT\dataset\IM_DA_TK_ALL.csv",
    parse_dates=['DATETIME']
)

df_tk = df_tk.rename(columns={
    'DATETIME': 'ds',
    'IM_TK': 'load'     # ค่าไฟ Tokyo
})
df_tk['city'] = 'Tokyo'

# ========== STEP 2: รวม + ทำ series ให้ต่อเนื่องทุก 30 นาที ==========
df = pd.concat(
    [df_ks[['ds', 'load', 'city']], df_tk[['ds', 'load', 'city']]],
    ignore_index=True
)

def make_regular_30min(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('ds')
    g = g.drop_duplicates(subset='ds')
    full_idx = pd.date_range(g['ds'].min(), g['ds'].max(), freq='30min')
    g = g.set_index('ds').reindex(full_idx)
    g['city'] = g['city'].ffill().bfill()
    g['load'] = g['load'].interpolate(limit_direction='both')
    g = g.reset_index().rename(columns={'index': 'ds'})
    return g

df = (
    df[['ds', 'load', 'city']]
    .groupby('city', group_keys=False)
    .apply(make_regular_30min)
)
df['month'] = df['ds'].dt.to_period('M')
df['date'] = df['ds'].dt.date

print("columns หลังจัดรูป:", df.columns.tolist())
print(df.head())

# ========== STEP 3: กำหนดช่วงวันสำหรับ Day-ahead evaluation ==========
first_date = df['date'].min()
last_date = df['date'].max()
start_eval_date = first_date + timedelta(days=1)

print("ช่วงวันที่ใช้ evaluate day-ahead:", start_eval_date, "→", last_date)

# ========== STEP 4: Loop ทำ Day-ahead 48-step per city per day ==========
all_preds = []

freq = '30min'  # 30 นาที
h = 48          # 48 จุดล่วงหน้า = 1 วัน

for city in df['city'].unique():
    city_df = df[df['city'] == city].sort_values('ds').reset_index(drop=True)
    city_dates = sorted(city_df['date'].unique())

    # เลือกเฉพาะวันที่อยู่ในช่วง eval และไม่ใช่วันแรกสุด (เพราะไม่มี history ก่อนหน้า)
    eval_dates = [d for d in city_dates
                  if (d >= start_eval_date) and (d != city_dates[0])]

    print(f"\n=== เมือง {city} ===")
    print("จำนวนวันที่จะพยากรณ์แบบ day-ahead:", len(eval_dates))

    for target_date in eval_dates:
        # history = ข้อมูลก่อนวัน target_date ทั้งหมด
        history = city_df[city_df['ds'] < pd.Timestamp(target_date)]
        if history.empty:
            continue

        # เตรียม df ให้ TimeGPT
        hist_tgpt = history[['ds', 'load']].copy()
        hist_tgpt['unique_id'] = city
        hist_tgpt = hist_tgpt[['unique_id', 'ds', 'load']].rename(
            columns={'load': 'y'}
        )

        # เรียก TimeGPT ให้พยากรณ์ 48 จุดของ "วัน target_date"
        fcst = nixtla_client.forecast(
            df=hist_tgpt,
            h=h,
            freq=freq,
            time_col='ds',
            target_col='y',
            id_col='unique_id',
            model='timegpt-1-long-horizon'
        )

        fcst = fcst.rename(columns={'TimeGPT': 'y_pred'})

        # ดึงค่าจริงของวัน target_date (48 จุด)
        actual_next = city_df[city_df['date'] == target_date][['ds', 'load']].copy()
        actual_next = actual_next.sort_values('ds')

        # รวม forecast + actual ด้วย ds
        merged = actual_next.merge(
            fcst[['ds', 'y_pred']],
            on='ds',
            how='inner'
        )
        if merged.empty:
            # ถ้า ds ไม่ตรงกัน (เช่น freq ไม่ match) ข้ามไป
            print(f"⚠ ข้ามวันที่ {target_date} ของเมือง {city} เพราะ merge ไม่เจอ ds ตรงกัน")
            continue

        merged['city'] = city
        merged['forecast_date'] = pd.Timestamp(target_date)
        merged['month'] = merged['ds'].dt.to_period('M')

        all_preds.append(merged)

# ========== STEP 5: รวมผล และคำนวณ metric ==========
if not all_preds:
    raise ValueError("ไม่มีผล day-ahead ถูกเก็บ (อาจเพราะ ds/freq ไม่ตรง)")

preds_df = pd.concat(all_preds, ignore_index=True)

def calc_metrics(g):
    err = g['load'] - g['y_pred']
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(
        np.abs(err) / np.where(g['load'] == 0, np.nan, g['load'])
    ) * 100
    return pd.Series({'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

# 5.1 metric รวมต่อเมือง (ทุกวันที่ evaluate)
metrics_by_city = (
    preds_df
    .groupby('city')
    .apply(calc_metrics)
    .reset_index()
)

print("\n=== Day-ahead 48-step metrics (รวมทุกวัน) ===")
print(metrics_by_city)

# 5.2 metric รายเดือน (ถ้าช่วง evaluate อยู่หลายเดือนจะเห็นหลายแถว)
metrics_by_city_month = (
    preds_df
    .groupby(['city', 'month'])
    .apply(calc_metrics)
    .reset_index()
)

print("\n=== Day-ahead 48-step metrics รายเดือน ===")
print(metrics_by_city_month)

# 5.3 metric รายวัน (ดูว่าแต่ละวันพลาดเยอะแค่ไหน)
metrics_by_city_day = (
    preds_df
    .groupby(['city', 'forecast_date'])
    .apply(calc_metrics)
    .reset_index()
)

print("\n=== Day-ahead 48-step metrics รายวัน (ตัวอย่าง 10 วันท้าย) ===")
print(metrics_by_city_day.sort_values(['city', 'forecast_date']).tail(10))