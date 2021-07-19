import os
import random
import csv
import test_queries_parser
from test_metrics_calculations import get_metrics


def find_result_calculated_score(result, records):
    for record in records:
        if int(record['doc_number']) == result:
            return test_queries_parser.calculated_score(record['raw_score'])
    return "-"


def find_result_original_score(result, records):
    for record in records:
        if int(record['doc_number']) == result:
            return record['raw_score']
    return "-"


def get_results_table_with_original_ranks(q, your_results, writer):
    title = ['query number', q["number"], 'query text', q["text"]]
    writer.writerow(title)
    writer.writerow([])
    header = ['Your order', 'doc id', 'original score', 'calculated score']
    writer.writerow(header)

    for i, result in enumerate(your_results):
        writer.writerow([i, result, find_result_original_score(result, q["records"]),
                         find_result_calculated_score(result, q["records"])])


def get_your_results(q):
    with open('vsm_inverted_index.json', 'r'):
        q = q['text']
        os.system(f'python vsm_ir.py query "{os.getcwd()}/vsm_inverted_index.json" "{q}"')
        with open('ranked_query_docs.txt', 'r') as results_file:
            return [int(i) for i in results_file]


def get_all_queries_metrics(queries, writer):
    header = ['query number', 'recall', 'precision', 'f_measuere']
    writer.writerow(header)

    for q in queries:
        your_results = get_your_results(q)
        recall, precision, f_measuere = get_metrics(q, your_results)
        current = [q['number'], recall, precision, f_measuere]
        writer.writerow(current)


def calculate_threshold(queries, writer):
    range_array = [i for i in range(0, 500, 10)]
    header = ['query number', 'max limit'] + [str(i) for i in range_array]
    writer.writerow(header)
    max_limits = []

    for q in queries:
        all_your_results = get_your_results(q)
        results = []
        for i in range_array:
            your_results = all_your_results[0: i]
            recall, precision, f_measuere = get_metrics(q, your_results)
            results.append(f_measuere)
        max_index = results.index(max(results))*10
        max_limits.append(max_index)
        current = [q['number']] + [max_index] + [x for x in results]
        writer.writerow(current)

    print(f'To get maximum F-measure stop retrieving docs after {sum(max_limits) / len(max_limits)}\n')


def controller(query_number=None):
    queries = test_queries_parser.parse_queries('cfc-xml_corrected')

    if query_number is None:
        query_number = random.randint(0, len(queries))

    query = queries[query_number]
    your_results = get_your_results(query)
    with open('random_query_status.csv', 'w') as random_query_status_file:
        writer = csv.writer(random_query_status_file)
        get_results_table_with_original_ranks(query, your_results, writer)

    with open('get_all_queries_metrics.csv', 'w') as all_queries_metrics_file:
        writer = csv.writer(all_queries_metrics_file)
        get_all_queries_metrics(queries, writer)

    with open('calculate_threshold.csv', 'w') as all_queries_metrics_file:
        writer = csv.writer(all_queries_metrics_file)
        calculate_threshold(queries, writer)

    print('finished tester')


controller()

