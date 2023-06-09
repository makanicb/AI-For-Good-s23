from sklearn.ensemble import RandomForestClassifier

def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = table[table[target] == target_value]
  e_list = t_subset[evidence]
  return sum([v==evidence_value for v in e_list])/len(e_list) + 0.01 #I skipped the ternary operator because sum will treat true=1 and false=0 by default

def cond_probs_product(table, evidence_values, target_column, target_val):
  evidence_columns = table.columns[:-1]
  evidence_complete = zip(evidence_columns, evidence_values)
  cond_prob_list = [cond_prob(table, e, ev, target_column, target_val) for e, ev, in evidence_complete]
  return up_product(cond_prob_list)

def prior_prob(table, target, target_value):
  t_list = table[target]
  return sum([v==target_value for v in t_list])/len(t_list)

def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the produce of the list, finally multiply by P(Flu=0)
  p_f0 = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)

  #do same for P(Flu=1|...)
  p_f1= cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)

  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(p_f0, p_f1)
  
  #return your 2 results in a list
  return [neg, pos]

def metrics (pred_act):
  assert isinstance(pred_act, list), f"Expected pred_act to be a list but found {type(pred_act)}."
  assert all([isinstance(i, list) for i in pred_act]), f"Expected pred_act to be a list of lists."
  assert all([len(i)==2 for i in pred_act]), f"Expected pred_act to be a list of pairs"
  assert all([all([isinstance(j, (int,float)) for j in i]) for i in pred_act]), f"Expected pred_act to be a list of pairs of ints"
  assert all([all([j >= 0 for j in i]) for i in pred_act]), f"Expected values of pred_act to be greater than equal to zero"
  #print(pred_act)
  tn = sum([list(pair)==[0,0] for pair in pred_act])
  tp = sum([list(pair)==[1,1] for pair in pred_act])
  fn = sum([list(pair)==[0,1] for pair in pred_act])
  fp = sum([list(pair)==[1,0] for pair in pred_act])
  #print (f"{tn=} {tp=} {fn=} {fp=}")
  accuracy = ((tn + tp) / (tn + tp + fn + fp)) if (tn + tp + fn + fp) > 0 else 0
  precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
  recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
  f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
  return {"Precision":precision, "Recall":recall, "F1":f1, "Accuracy":accuracy}

def try_archs(full_table, target, architectures, thresholds):
  train_table, test_table = up_train_test_split(full_table, target, .4)

  #copy paste code here
  for arch in architectures:
    all_results= up_neural_net(train_table, test_table, arch, target)
    #loop through thresholds
    all_mets = []
    for t in thresholds:
      all_predictions = [1 if pos>=t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]


    print(f'Architecture: {arch}')
    print(up_metrics_table(all_mets))

  return None  #main use is to print out threshold tables, not return anything useful.

def run_random_forest(train, test, target, n):

  #your code below
  X = up_drop_column(train, target)
  y = up_get_column(train,target)
  k_feature_table = up_drop_column(test, target) 
  k_actuals = [int(a) for a in up_get_column(test, target)]
  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)  
  clf.fit(X, y)
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for n,p in probs]  #probs is list of [neg,pos] like we are used to seeing.
  pos_probs[:5]
  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(all_predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  all_mets[:2]
  metrics_table = up_metrics_table(all_mets)
  print(metrics_table)  #output we really want - to see the table
  return None
