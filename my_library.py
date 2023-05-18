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
