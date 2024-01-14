from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import pandas as pd
from functions import extract_year, year_pipe_names_out
import itertools


class LoanApplication(BaseModel):
    loan_amnt: float
    term: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    title: str
    dti: float
    delinq_2yrs: float
    fico_range_high: float
    inq_last_6mths: float
    mths_since_last_delinq: float
    mths_since_last_record: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    initial_list_status: str
    mths_since_last_major_derog: float
    dti_joint: float
    tot_coll_amt: float
    tot_cur_bal: float
    open_acc_6m: float
    open_act_il: float
    open_il_12m: float
    open_il_24m: float
    mths_since_rcnt_il: float
    total_bal_il: float
    il_util: float
    open_rv_12m: float
    open_rv_24m: float
    max_bal_bc: float
    all_util: float
    total_rev_hi_lim: float
    inq_fi: float
    total_cu_tl: float
    inq_last_12m: float
    acc_open_past_24mths: float
    bc_open_to_buy: float
    bc_util: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    mths_since_recent_bc: float
    mths_since_recent_bc_dlq: float
    mths_since_recent_inq: float
    mths_since_recent_revol_delinq: float
    num_accts_ever_120_pd: float
    num_actv_bc_tl: float
    num_actv_rev_tl: float
    num_bc_sats: float
    num_bc_tl: float
    num_il_tl: float
    num_op_rev_tl: float
    num_rev_accts: float
    num_tl_90g_dpd_24m: float
    num_tl_op_past_12m: float
    pct_tl_nvr_dlq: float
    percent_bc_gt_75: float
    pub_rec_bankruptcies: float
    total_bal_ex_mort: float
    total_bc_limit: float
    sec_app_fico_range_high: float
    disbursement_method: str
    earliest_cr_line: str


class Grade(BaseModel):
    grade: str


class SubGrade(BaseModel):
    subgrade: str
    int_rate: float


grade_order = ["G", "F", "E", "D", "C", "B", "A"]
subgrade_numbers = ["5", "4", "3", "2", "1"]
combinations = list(itertools.product(grade_order, subgrade_numbers))
subgrade_order = ["".join(combination) for combination in combinations]


# all preprocessors are the same
preprocessor = joblib.load("../models/grade_preprocessor_V1.joblib")
grade_model = joblib.load("../models/grade_model_V1.joblib")
subgrade_model = joblib.load("../models/subgrade_model_V1.joblib")
int_rate_model = joblib.load("../models/int_rate_model_V1.joblib")


app = FastAPI()


@app.post("/predict_grade", response_model=Grade)
def predict_grade(payload: LoanApplication):
    df = pd.DataFrame([payload.model_dump()])
    df_tf = preprocessor.transform(df)
    grade_num = int(grade_model.predict(df_tf))
    grade = grade_order[grade_num]
    result = dict(grade=grade)
    return result


@app.post("/predict_subgrade", response_model=SubGrade)
def predict_subgrade(payload: LoanApplication):
    df = pd.DataFrame([payload.model_dump()])
    df_tf = preprocessor.transform(df)
    subgrade_num = int(subgrade_model.predict(df_tf))
    subgrade = subgrade_order[subgrade_num]
    int_rate = int_rate_model.predict(df_tf)
    result = dict(subgrade=subgrade, int_rate=int_rate)
    return result
