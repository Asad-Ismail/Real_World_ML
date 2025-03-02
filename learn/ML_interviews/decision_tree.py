
import math

def counter(items:list)->dict:
	count={}
	for item in items:
		count[item]=count.get(item,0)+1
	return count


def entropy(labels):
	counts=counter(labels)
	ent = -sum( (v/len(labels)) * math.log2(v/len(labels)) for v in counts.values())
	return ent

def get_best_feature(examples,attributes, target_attr):
	
	ig={}
	target_values=[item[target_attr] for item in examples]
	total_entropy= entropy(target_values)
	for attr in attributes:
		unique_attr_values = set(item[attr] for item in examples)
		weighted_entropy=0
		for attr_value in unique_attr_values:
			subset_examples = [item for item in examples if item[attr]==attr_value] 
			subset_targets = [item[target_attr] for item in subset_examples]
			weighted_entropy +=(len(subset_examples)/len(examples))*entropy(subset_targets)
		ig[attr]= total_entropy-weighted_entropy
	return max(ig,key=ig.get)


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
	
	target_values =[item[target_attr] for item in examples]
	if len(set(target_values))==1:
		return target_values[0]
	counts = counter(target_values) 
	if not attributes:
		return max(counts, key=counts.get)
	
	best_feature = get_best_feature(examples, attributes, target_attr)

	decision_tree ={best_feature:{}}

	unique_values = set(item[best_feature] for item in examples)
	for value in unique_values:
		subset=[item for item in examples if item[best_feature]==value]
		if not len(subset):
			return max(counts, key=counts.get)
		new_features= [item for item in attributes if item!=best_feature]
		subtree = learn_decision_tree(subset,new_features,target_attr)
		decision_tree[best_feature][value] = subtree

	return decision_tree