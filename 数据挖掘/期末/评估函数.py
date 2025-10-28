def calculate_recall_at_k(result, gold, k):
    result = read_json(result)
    gold = read_json(gold)
    total_queries = len(result)
    recall = 0
    for r,g in zip(result, gold):
        correct_queries = 0
    #去除开始，结尾字
        relevant_facts = [e['fact_input_list'][1:-1] for e in g['evidence_list']]
        top_k_facts = [e['fact_input_list'][1:-1] for e in r['evidence_list'][:k]]
        #转换成字符明
        relevant_facts =[''.join(map(str, sublist))for sublist in relevant_facts]
        top_k_facts =[''.join(map(str, sublist)) for sublist in top_k_facts]
        if relevant_facts: # 检査relevant_facts是否为空
            for answer in relevant_facts:
                if any(answer in fact for fact in top_k_facts):
                    correct_queries += 1
                    recall += correct_queries /len(relevant_facts) # 使用relevant_facts的长度米归一
    recall_at_k = recall / total_queries
    return recall_at_k

def calculate_mrr_at_k(result, gold, k):
    result = read_json(result)
    gold = read_json(gold)
    total_queries = len(result)
    mrr = 0
    mrr_at_k = 0
    for r, g in zip(result, gold):
        reciprocal_ranks = []
        relevant_facts = [e['fact_input_list'][1:-1] for e in g['evidence_list']]
        top_k_facts = [e['fact_input_list'][1:-1] for e in r['evidence_list'][:k]]
        relevant_facts = [''.join(map(str, sublist)) for sublist in relevant_facts]
        top_k_facts = [''.join(map(str, sublist)) for sublist in top_k_facts]
        if relevant_facts:  # 检査relevant_facts是否为空
            for answer in relevant_facts:
                for i, fact in enumerate(top_k_facts):
                    if answer in fact:
                        reciprocal_ranks.append(1 / (i + 1))  # 计算倒数排名并添加到列表中
                        break
            mrr += sum(reciprocal_ranks) / len(relevant_facts)
    mrr_at_k = mrr / total_queries
    return mrr_at_k