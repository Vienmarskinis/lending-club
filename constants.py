import itertools


# All constants regarding accepted loans dataset manipulation


emp_length_order = [
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]


emp_length_order_eda = emp_length_order
emp_length_order_eda.append("None")


wrong_str_to_cast = [
    "revol_bal_joint",
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
]


verification_order = ["Not Verified", "Verified", "Source Verified"]


grade_order = ["G", "F", "E", "D", "C", "B", "A"]
subgrade_numbers = ["5", "4", "3", "2", "1"]
combinations = list(itertools.product(grade_order, subgrade_numbers))
subgrade_order = ["".join(combination) for combination in combinations]


binary_cols = ["term", "initial_list_status", "application_type", "disbursement_method"]
high_cardinal_cols = ["emp_title", "title", "zip_code", "addr_state", "purpose"]
ordinal_cols = ["emp_length", "verification_status", "verification_status_joint"]
nominal_cols = ["home_ownership"]
target_cols = ["grade", "sub_grade", "int_rate"]


high_corr_to_drop = [
    "sec_app_fico_range_low",
    "fico_range_low",
    "num_sats",
    "num_rev_tl_bal_gt_0",
    "tot_hi_cred_lim",
    "total_il_high_credit_limit",
    "avg_cur_bal",
]
boruta_to_drop = [
    "collections_12_mths_ex_med",
    "annual_inc_joint",
    "acc_now_delinq",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "tax_liens",
    "revol_bal_joint",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
    "issue_d",
    "sec_app_earliest_cr_line",
    "application_type",
    "emp_title",
    "zip_code",
    "addr_state",
    "emp_length",
    "verification_status_joint",
]
to_drop_in_pipe = "home_ownership_infrequent_sklearn"


wrong_str_to_cast_selected = list(
    set(wrong_str_to_cast) - set(boruta_to_drop) - set(high_corr_to_drop)
)
binary_cols_selected = list(
    set(binary_cols) - set(boruta_to_drop) - set(high_corr_to_drop)
)
high_cardinal_cols_selected = list(
    set(high_cardinal_cols) - set(boruta_to_drop) - set(high_corr_to_drop)
)
ordinal_cols_selected = list(
    set(ordinal_cols) - set(boruta_to_drop) - set(high_corr_to_drop)
)
nominal_cols_selected = list(
    set(nominal_cols) - set(boruta_to_drop) - set(high_corr_to_drop)
)
