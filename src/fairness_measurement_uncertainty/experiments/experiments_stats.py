import pandas as pd
import scipy.stats as stats


def assumption_test(df: pd.DataFrame, assumption: int, sens_value: int) -> float:
    """
    Returns the
    :param df: input data with columns "A", "Ahat", "S"
    :param assumption: 1: first worldview, 2: second worldview
    :param sens_value: Value of the sensitive attribute
    :return: maximum p_val of the two tests
    """

    res_str = ""
    if assumption == 1:
        # test Ahat <- A -> S
        # ANOVA test
        # res1 = stats.f_oneway(df[(df['Ahat'] == sens_value) & (df['A'] == sens_value)]['S'],
        #                       df[(df['Ahat'] != sens_value) & (df['A'] == sens_value)]['S'])
        # res2 = stats.f_oneway(df[(df['Ahat'] == sens_value) & (df['A'] != sens_value)]['S'],
        #                       df[(df['Ahat'] != sens_value) & (df['A'] != sens_value)]['S'])

        # Kruskal-Wallis H Tes
        res1 = stats.kruskal(df[(df['Ahat'] == sens_value) & (df['A'] == sens_value)]['S'],
                              df[(df['Ahat'] != sens_value) & (df['A'] == sens_value)]['S'])
        res2 = stats.kruskal(df[(df['Ahat'] == sens_value) & (df['A'] != sens_value)]['S'],
                              df[(df['Ahat'] != sens_value) & (df['A'] != sens_value)]['S'])
    elif assumption == 2:
        # test A -> Ahat -> S
        # ANOVA test
        # res1 = stats.f_oneway(df[(df['A'] == sens_value) & (df['Ahat'] == sens_value)]['S'],
        #                       df[(df['A'] != sens_value) & (df['Ahat'] == sens_value)]['S'])
        # res2 = stats.f_oneway(df[(df['A'] == sens_value) & (df['Ahat'] != sens_value)]['S'],
        #                       df[(df['A'] != sens_value) & (df['Ahat'] != sens_value)]['S'])

        # Kruskal-Wallis H Tes
        res1 = stats.kruskal(df[(df['A'] == sens_value) & (df['Ahat'] == sens_value)]['S'],
                              df[(df['A'] != sens_value) & (df['Ahat'] == sens_value)]['S'])
        res2 = stats.kruskal(df[(df['A'] == sens_value) & (df['Ahat'] != sens_value)]['S'],
                              df[(df['A'] != sens_value) & (df['Ahat'] != sens_value)]['S'])
    else:
        raise ValueError(f"Unsupported assumption: {assumption}")

    res = max(res1[1], res2[1])

    return res
