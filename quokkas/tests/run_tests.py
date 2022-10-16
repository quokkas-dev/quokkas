from generic.dataframe_functionality_test import test_create_df, test_column_operations

tests = [test_create_df, test_column_operations]


if __name__ == "__main__":
    successful_count = 0
    failed_tests = []
    for i, test in enumerate(tests):
        if test():
            successful_count += 1
        else:
            failed_tests.append(test.__name__)
    print(f'{successful_count} tests out of {len(tests)} successful, {len(tests) - successful_count} tests failed. ')
    if successful_count != len(tests):
        print('Failed tests:')
        print(failed_tests)

