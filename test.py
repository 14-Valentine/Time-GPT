import pandas as pd
import numpy as np
from nixtla import NixtlaClient

# ========== ตั้งค่า TimeGPT client ==========
# ใส่ API key ของคุณเองแทน xxx
nixtla_client = NixtlaClient(
    api_key="nixak-EDZ03wsobkQu2aMjPLjRTpJCII7aWlcrgnS0WO7DNT0yUQxdIB4g8GU2lQKPrWs8gMe19YSWTVnlF97F"
)
# ถ้าต้องการเช็คว่า key ใช้ได้ไหม
# print(nixtla_client.validate_api_key())

# ========== STEP 1: อ่านและรวม 2 ไฟล์ ==========
# Kansai
df_ks = pd.read_csv(
    r"D:\Time-GPT\dataset\IM_DA_KS_ALL.csv",
    parse_dates=['DATE']
)

df_ks = df_ks.rename(columns={
    'DATE': 'ds',               # คอลัมน์เวลา
    'IM DA KS ALL IM': 'load'   # <<< ตรวจให้ตรงกับหัวคอลัมน์จริงในไฟล์ Kansai
})
df_ks['city'] = 'Kansai'

# Tokyo
df_tk = pd.read_csv(
    r"D:\Time-GPT\dataset\IM_DA_TK_ALL.csv",
    parse_dates=['DATE']
)

df_tk = df_tk.rename(columns={
    'DATE': 'ds',               # คอลัมน์เวลา
    'DA TK ALL': 'load'         # <<< ถ้าหัวจริงคือ 'IM DA TK ALL' ให้แก้เป็นชื่อนั้น
})
df_tk['city'] = 'Tokyo'

# รวมสองเมือง
df = pd.concat([df_ks, df_tk], ignore_index=True)
df = df.sort_values(['ds', 'city'])

# ========== STEP 2: เตรียมคอลัมน์ month ==========
df['month'] = df['ds'].dt.to_period('M')

# ========== STEP 3: แยก train / test ==========
# train: 2022-01-01 ถึง 2023-12-31
train = df[(df['ds'] >= '2022-01-01') & (df['ds'] <= '2023-12-31')]
# test: ข้อมูลหลังจากนั้น (ใช้ไว้เช็คความแม่น)
test = df[df['ds'] > '2023-12-31']

if test.empty:
    raise ValueError("ยังไม่มี test data หลัง 2023-12-31 ในไฟล์ของคุณ")

# ========== STEP 4: เตรียม data ใน format ที่ TimeGPT ชอบ ==========
# TimeGPT by default ต้องการคอลัมน์: unique_id, ds, y
train_tgpt = train[['ds', 'city', 'load']].rename(
    columns={'city': 'unique_id', 'load': 'y'}
)

test_tgpt = test[['ds', 'city', 'load']].rename(
    columns={'city': 'unique_id', 'load': 'y'}
)

# ตรวจว่าจำนวนจุดใน test ต่อเมืองเท่ากันไหม (จะ forecast เท่ากันทุกเมือง)
test_sizes = test_tgpt.groupby('unique_id').size()
print("จำนวนจุดใน test ต่อเมือง:\n", test_sizes)

if test_sizes.nunique() != 1:
    raise ValueError("ตอนนี้ตัวอย่างสมมติว่าแต่ละเมืองมีจำนวนจุด test เท่ากัน ถ้าไม่เท่ากันต้องแยกเรียก TimeGPT ต่อเมือง")

h = int(test_sizes.iloc[0])   # horizon = จำนวนจุดใน test ต่อเมือง
print(f"ใช้ horizon h = {h} จุด (เท่ากับความยาวช่วง test ต่อเมือง)")

# ความถี่ 30 นาที ใช้ freq='30T' ตาม pandas
freq = '30T'

# ========== STEP 5: เรียก TimeGPT forecast ==========
timegpt_fcst = nixtla_client.forecast(
    df=train_tgpt,
    h=h,
    freq=freq,
    time_col='ds',
    target_col='y',
    id_col='unique_id',
    # optional: finetune_steps=20, finetune_loss='mae',
    model='timegpt-1'  # หรือ 'timegpt-1-long-horizon' ถ้า horizon ยาวมาก
)

# ผลลัพธ์จะมีคอลัมน์ประมาณ: ['unique_id', 'ds', 'TimeGPT', ...]
print(timegpt_fcst.head())

# เปลี่ยนชื่อคอลัมน์ให้พร้อม merge กับ test
timegpt_fcst = timegpt_fcst.rename(columns={
    'unique_id': 'city',
    'TimeGPT': 'y_pred'   # ชื่อคอลัมน์พยากรณ์จาก TimeGPT
})

# ========== STEP 6: เอา forecast ไปเทียบกับ test แล้วคำนวณ metric ==========
test_eval = test.copy()   # ['ds', 'city', 'load', 'month', ...]
test_eval = test_eval.merge(
    timegpt_fcst[['city', 'ds', 'y_pred']],
    on=['city', 'ds'],
    how='inner'
)

if test_eval.empty:
    raise ValueError("ผล forecast จาก TimeGPT ไม่ตรงช่วงเวลา/เมืองกับ test ตรวจ freq และช่วงเวลา train/test อีกที")

# คำนวณ MAE / RMSE / MAPE รายเดือน แยกเมือง
def calc_metrics(g):
    err = g['load'] - g['y_pred']
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    # กันหารศูนย์
    mape = np.mean(np.abs(err) / np.where(g['load'] == 0, np.nan, g['load'])) * 100
    return pd.Series({'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

monthly_results = (
    test_eval
    .groupby(['city', test_eval['ds'].dt.to_period('M')])
    .apply(calc_metrics)
    .reset_index()
    .rename(columns={'ds': 'month'})
)

print(monthly_results)