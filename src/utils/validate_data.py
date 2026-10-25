import great_expectations as ge
from typing import Tuple, List

def validate_telco_data(df):
    ge_df = ge.dataset.PandasDataset(df)
    ge_df.expect_column_to_exist("customerID")
    ge_df.except_column_values_to_not_be_null("customerID")
    ge_df.except_column_to_exist("gender")
    ge_df.except_column_to_exist("Partner")
    ge_df.except_column_to_exist("Dependents")
    ge_df.except_column_to_exist("PhoneService")
    ge_df.except_column_to_exist("InternetService")
    ge_df.except_column_to_exist("Contract")
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")
    ge_df.expect_column_to_be_in_set("gender", ["Male", "Female"])
    ge_df.expect_column_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_to_be_in_set("PhoneService", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])
    ge_df.expect_column_values_to_be_in_set("InternetService", ["DSL", "Fiber optic", "No"])
    ge_df.expect_column_values_to_be_between("tenure", min_value = 0)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value = 0)
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    ge_df.expect_column_pair_values_A_to_be_grater_than_B(
        column_A = "TotalCharges",
        column_B = "MonthlyCharges",
        or_equal = True,
        mostly = 0.95
    )

    results = ge_df.validate()

    failed_expectations = []

    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)

    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successfully")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")

    return results["success"], failed_expectations