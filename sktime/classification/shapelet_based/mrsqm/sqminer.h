/*
 * sqminer.h
 *
 *  Created on: 3 Jul 2017
 *      Author: thachln
 */

#ifndef SQMINER_H_
#define SQMINER_H_

#include <vector>
#include <map>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <set>
#include <map>

#include "common.h"

#define NULL nullptr

using namespace std;

template <typename T1, typename T2>
struct pair_2nd_cmp_alt: public std::binary_function<bool, T1, T2> {
	bool operator () (const std::pair <T1, T2>& x1, const std::pair<T1, T2> &x2)
	{
		return x1.second > x2.second;
	}
};

class ENode{
private:

	char first_char = 'a';
	

	void initialize_enode(string _ngram, int max_children){
		children.resize(max_children);
		std::fill(children.begin(), children.end(), NULL);
		lastdocid = 0;
		ngram = _ngram;
		selected = false;
		foot_print_covered = false;


	}
public:
	const static int MAX_CHILDREN = 26;
	string ngram;

	vector<int> loc;
	int lastdocid;
	vector<ENode*> children;

	bool selected;

	bool foot_print_covered;




	double chi_square;
	double bound;


	int external_index; // index of the feature in the feature space

	ENode(){
		initialize_enode("",MAX_CHILDREN);
	}

	ENode(int max_children){
		initialize_enode("",max_children);
	}

	ENode(char first_char, int max_children){
		this->first_char = first_char;
		initialize_enode("",max_children);
	} 

	~ENode(){
		for(auto c: children){
			delete c;
		}
	}

	// docid starts at 1
	void add_location(int docid, int position){
		docid += 1;
		if (docid != lastdocid){
			loc.push_back(-docid);
			lastdocid = docid;
		}
		loc.push_back(position);
	}



	ENode* get_child(char c){
		int index = c - first_char;
		if (index < children.size()){
			if (children[index] == NULL){
				children[index] = new ENode(first_char, children.size());
				children[index]->ngram = ngram + c;
				//children[index]->best = best;

			}
			return children[index];
		}
		return NULL;
	}

	ENode* get_child_without_creating(char c){
		int index = c - first_char;
		return children[index];
	}

	ENode* copy(){
		ENode* node = new ENode(first_char, children.size());

		node->selected = selected;
		node->chi_square = chi_square;
		node->external_index = external_index;

		for (int i = 0; i < MAX_CHILDREN; i++){
			if (children[i] != NULL){
				node->children[i] = children[i]->copy();
			}
		}

		return node;
	}



	// print information of this node
	void print(){

		for(int i = 0; i < loc.size(); i++){
			cout << loc[i] << " ";
		}
		cout << endl;
	}

	// recursive print child nodes
	void print_r(){
		print();
		for(auto child: children){
			if (child != NULL){
				child->print_r();
			}
		}
	}
};


// rank and store the top nodes only
// capacity = -1 to store all the nodes
class NodeStore{
private:
	vector<ENode*> store;
	int capacity;
public:
	NodeStore(int capacity){
		this->capacity = capacity;
	}

	bool insert_node(ENode *node_to_insert){

		if (node_to_insert->foot_print_covered){
			return false;
		}


		if (capacity != -1){
			auto it = upper_bound(store.begin(), store.end(), node_to_insert,
					[](ENode *lhs, ENode *rhs ) { return lhs->chi_square > rhs->chi_square;});
			store.insert(it, node_to_insert);
			node_to_insert->selected = true;
			while(store.size() > capacity){
				store.back()->selected = false;
				store.pop_back();
			}
		} else { // unlimited (no need for sorting)
			node_to_insert->external_index = store.size();
			store.push_back(node_to_insert);
		}

		return true;
	}

	double threshold(){
		if ((capacity != -1) && (store.size() == capacity)){
			return store.back()->chi_square;
		}
		else {
			return 0.0;
		}
	}

	int size(){
		return store.size();
	}

	ENode* get_node(int i){
		return store[i];
	}

	void clear(){
		store.clear();
	}

	void print(){
		for (ENode* node: store){
			cout << node->ngram << " : " << node->chi_square << endl;
		}
	}
};



class SQMiner{
private:

	// store class manager
	class LabelManager{
	public:
		vector<int> unique_y;
		vector<double> y_prob;
		vector<int> y;

		LabelManager(vector<int>& labels){
			vector<int> frequency;
			for (int lb: labels){
				int id = get_unique_label_index(lb);
				if (id < 0){
					unique_y.push_back(lb);
					frequency.push_back(1);
				} else {
					frequency[id]++;
				}
			}
			y_prob.resize(unique_label_count());

			for (int i = 0; i < unique_label_count(); i++){
				y_prob[i] = frequency[i] * 1.0 / labels.size();
			}

			// copy label vector
			y = labels;
		}

		int get_label(int index){
			return y[index];
		}

		int unique_label_count(){
			return unique_y.size();
		}

		int get_unique_label_index(int label){

			for (int i = 0; i < unique_label_count(); i++){
				if (unique_y[i] == label){
					return i;
				}
			}
			return -1; // return value if index not found
		}

		// locs stored by ENode thus the id is negative
		vector<int> count_from_locs(vector<int> &locs){
			vector<int> count (unique_label_count(), 0);

			for (int l: locs){
				if (l < 0){
					int lb = y[-l - 1];
					//cout << lb << endl;
					count[get_unique_label_index(lb)]++;
				}
			}
			return count;
		}


	};

	vector<vector<int>> ft_matrix;

	int number_of_ft = 0;

	NodeStore store = NodeStore(-1);

	// store root for using later
	ENode* root;
	//vector<string> sequences;
	//vector<int> y;
	LabelManager* ymgr;

	char first_char = 'a';
	int alphabet_size = 26;

	double selection;


	// prepare inverted index of all unigrams (unigram -> location)
	ENode* prepare_inverted_index(vector<string> &sequences, vector<int>& y){
		ENode* root = new ENode(first_char,alphabet_size);
		for (int i = 0; i < sequences.size(); i++){
			int cur_pos = 0;
			for (char & c : sequences[i]){
				if (!isspace(c)){
					root->get_child(c)->add_location(i,cur_pos);

				}
				cur_pos++;
			}
		}

		return root;
	}

	void compute_chi_square_score_and_bound(ENode* node){
		// count class frequency of this feature


		// compute chi square
		vector<int> observed = ymgr->count_from_locs(node->loc);
		node->chi_square = chi_square_score(observed, ymgr->y_prob);

		// compute bound
		node->bound = chi_square_bound(observed, ymgr->y_prob);



	}



	bool expand_node(ENode* node,vector<string> &sequences){
		int current_doc = -1;
		// expanding the current candidate
		bool child_found = false;
		for (auto pos: node->loc){
			if (pos < 0){
				current_doc = -pos - 1;
			} else {
				int next_pos = pos + 1;
				//while(next_pos < sequences[current_doc].size() && isspace(sequences[current_doc][next_pos])){
				//	next_pos++;
				//}
				if (next_pos < sequences[current_doc].size() && !isspace(sequences[current_doc][next_pos])){
					char next_unigram = sequences[current_doc][next_pos];
					ENode* next = node->get_child(next_unigram);
					next->add_location(current_doc,next_pos);
					child_found = true;
				}
			}
		}
		// compare node finger-print with its children
		if (node->selected || node->foot_print_covered){
		for (auto child: node->children){
			if (child != NULL){
				int ci = 0;
				int pi = 0;
				bool matched = false;
				while (child->loc[ci] == node->loc[pi]){
					ci++;
					pi++;
					while (child->loc[ci] > 0 && ci < (child->loc.size() - 1)){
						ci++;
					}
					while(node->loc[pi] > 0 && pi < (node->loc.size() - 1)){
						pi++;
					}

					if (pi == (node->loc.size() - 1) && ci == (child->loc.size() - 1)){
						matched = true;
						break;
					}

				}
				if (matched){
					// cout << "Found a matched: " << endl;
					// cout << node->ngram << " : ";
					// for (auto pos: node->loc){
					// 	cout << pos << " ";
					// }
					// cout << endl;
					// cout << child->ngram << " : ";
					// for (auto pos: child->loc){
					// 	cout << pos << " ";
					// }
					// cout << endl;
					child->foot_print_covered = true; // flag the child with the same fingerprint so it won't be selected later

				}
				

			}

		}
		}
		return child_found;
	}




	bool can_prune_with_store(ENode* node){

		compute_chi_square_score_and_bound(node);

		store.insert_node(node);

		if (node->bound <= store.threshold()){
			return true;
		}

		return false;
	}

	bool can_prune_with_pvalue(ENode* node, double p_threshold){		
		compute_chi_square_score_and_bound(node);
		double pvalue = chi2_pvalue(ymgr->unique_label_count() - 1,node->chi_square);
		// cout << node->ngram << ":" << node->chi_square << ":" << pvalue << endl;
		


		if (pvalue <= p_threshold){
			node->selected = true;
			// cout << "selected" << endl;
			store.insert_node(node);
		}



		if (chi2_pvalue(ymgr->unique_label_count() - 1,node->bound) > p_threshold){

			return true;
		}

		return false;
	}

	// never prune
	bool can_prune_mock(ENode* node){
		node->selected = true;
		store.insert_node(node);
		return false;
	}

	bool can_prune(ENode* node){
		if (selection <= 0) { // brute force
			return can_prune_mock(node);
		} else if (selection < 1) { // chi-squared test with p-value = selection
			return can_prune_with_pvalue(node, selection);
		} else { // top k sequences with k = int(selection)
			return can_prune_with_store(node);
		}

	}



public:
	SQMiner(){

	}

	SQMiner(int n_ssq){
		store = NodeStore(n_ssq);
	}

	SQMiner(char first_char, int alphabet_size, int n_ssq){
		store = NodeStore(n_ssq);
		this->first_char = first_char;
		this->alphabet_size = alphabet_size;

	}

	SQMiner(double selection){
		if (selection <= 0) { // brute force

		} else if (selection < 1) { // chi-squared test with p-value = selection

		} else { // top k sequences with k = int(selection)
			store = NodeStore(int(selection));
		}

		this->selection = selection;

	}

	void configure_alphabet(char first_char, int alphabet_size){
		this->first_char = first_char;
		this->alphabet_size = alphabet_size;
	}


	vector<string> mine(vector<string> &sequences, vector<int>& y){

		vector<string> output;
		ymgr = new LabelManager(y);

		if (ft_matrix.empty()){
			for (int i = 0; i < sequences.size(); i++){
				ft_matrix.push_back(vector<int>());
			}
			ft_matrix.shrink_to_fit();
		}


		set<ENode*> best_features;
		int node_count = 0;



		root = prepare_inverted_index(sequences, y);
		vector<ENode*> unvisited;
		for(auto child:root->children){
			if (child != NULL){
				unvisited.push_back(child);
			}
		}

		while(!unvisited.empty()){
			node_count++;
			ENode* current_node = unvisited.back();
			unvisited.pop_back();

			// if path cannot be pruned then expand the node
			if ((!can_prune(current_node)) && expand_node(current_node,sequences)){
				// add new candidates to unvisited list
				for(auto child:current_node->children){
					if (child != NULL){
						unvisited.push_back(child);
					}
				}
			}
		}
		// cout << "Nodes: " << node_count << endl;
		// cout << "Number of candidates: " << store.size() << endl;
		//store.print();



		// select a certain top features
		// TODO: need a smarted selection method
		//int selected = (int)sorted_features.size() * 0.03;

		// add new features to feature matrix with default value 0
		for (int i = 0; i < ft_matrix.size(); i++){
			ft_matrix[i].resize(number_of_ft + store.size());
			fill(ft_matrix[i].begin()+ number_of_ft,ft_matrix[i].end(),0);
		}
		// update feature vector
		for (int i = 0; i < store.size(); i++){
			store.get_node(i)->external_index = i;
			for (auto pos:store.get_node(i)->loc){
				if (pos < 0) {
					ft_matrix[-pos - 1][number_of_ft + i] = 1;
				}
			}
			//cout << "Feature: " << sorted_features[i].first << " " << sorted_features[i].second << endl;
			output.push_back(store.get_node(i)->ngram);
			//cout << store.get_node(i)->ngram << " : " << store.get_node(i)->chi_square << endl;
			//cout << i << " " << store.get_node(i)->external_index << " " << store.get_node(i)->selected << endl;
		}

		number_of_ft += store.size();

		//root->print_r();
		delete root;
		return output;
	}

	void write_mined_sequences(string path){
		std::ofstream writer (path);
		for (int i = 0; i < store.size(); i++){
			writer << store.get_node(i)->ngram << "," << store.get_node(i)->chi_square << endl;
		}
		writer.close();
	}

	void generate_test_data(vector<string> &sequences, vector<int> &labels, string path){
		// copy tree

		ENode* test_root = root->copy();

		// prepare inverted index

		vector<ENode*> unvisited;


		for (ENode* cn: test_root->children){
			if (cn != NULL){
				unvisited.push_back(cn);
			}
		}

		for (int i = 0; i < sequences.size(); i++){
			int cur_pos = 0;
			for (char & c : sequences[i]){
				if (!isspace(c)){
					ENode* cn = test_root->get_child_without_creating(c);
					if (cn != NULL){
						cn->add_location(i,cur_pos);
					}
				}
				cur_pos++;
			}
		}

		// explore tree
		vector<vector<int>> new_ft_matrix;
		for (int i = 0; i < sequences.size(); i++){
			new_ft_matrix.push_back(vector<int>(store.size(),0));
		}

		while(!unvisited.empty()){
			ENode* current_node = unvisited.back();
			unvisited.pop_back();

			// if node is a feature
			if (current_node->selected){
				for (auto pos:current_node->loc){
					if (pos < 0) {
						new_ft_matrix[-pos - 1][current_node->external_index] = 1;
						//cout << current_node->external_index << endl;
					}
				}
			}

			// expand node
			int current_doc = 0;
			for (auto pos: current_node->loc){
				if (pos < 0){
					current_doc = -pos - 1;
				} else {
					int next_pos = pos + 1;
					//while(next_pos < sequences[current_doc].size() && isspace(sequences[current_doc][next_pos])){
					//	next_pos++;
					//}
					if (next_pos < sequences[current_doc].size() && !isspace(sequences[current_doc][next_pos])){
						char next_unigram = sequences[current_doc][next_pos];
						ENode* next = current_node->get_child_without_creating(next_unigram);
						if (next != NULL){
							next->add_location(current_doc,next_pos);
						}

					}
				}
			}

			for (ENode* cn: current_node->children){
				if (cn != NULL){
					unvisited.push_back(cn);
				}
			}
		}

		// write to file
		std::ofstream writer (path);
		for (int i = 0; i < new_ft_matrix.size(); i++){
			// writer << labels[i];
			for (int f: new_ft_matrix[i]){
				writer << f << " ";
			}
			if (i < (new_ft_matrix.size() - 1)){
				writer << endl;
			}
		}
		writer.close();



		// remove tree
		delete test_root;



	}


	// only works with brute force
	void match_with_tree(vector<string> &sequences, vector<int> &labels, string path){
		//vector<vector<int>> ft_matrix;
		std::ofstream writer (path);
		for (int i = 0; i < labels.size();i++){
			vector<int> vt;
			vt.resize(store.size());
			fill(vt.begin(),vt.end(),0);
			writer << labels[i];

			string sequence = sequences[i];
			for (int c = 0; c < sequence.length(); c++){
				ENode *current = root;
				int cur_pos = c;
				while ((current != NULL) && (cur_pos < sequence.length()) && (!isspace(sequence.at(cur_pos)))){
					if (current != root){
						vt[current->external_index] = 1;
					}
					int next_node = sequence.at(cur_pos) - first_char;
					current = current->children[next_node];
					cur_pos++;
				}


			}

			for (int f: vt){
				writer << " " << f;
			}
			writer << endl;

		}
		writer.close();
	}

	void write_ft_matrix(string path, vector<int>& labels){
		std::ofstream writer (path);
		for (int i = 0; i < ft_matrix.size(); i++){
			// writer << labels[i];
			for (int f: ft_matrix[i]){
				writer << f << " ";
			}
			if (i < (ft_matrix.size() - 1)){
				writer << endl;
			}
		}
		writer.close();
	}



};


#endif /* SQMINER_H_ */
