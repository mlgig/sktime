#include "sqminer.h"

#include "common.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <ctime>

using namespace std;

void read_multi_data(string path, vector<vector<string>> &sequences, vector<int> &labels)
{
    ifstream in(path);

    std::string str;
    int last_config = -1;

    //bool print = true;

    while (std::getline(in, str))
    {
        // extract index
        int first_space = str.find_first_of(" ");
        int cr_config = stoi(str.substr(0, first_space));
        str = str.substr(first_space + 1);
        // extract class
        int second_space = str.find_first_of(" ");
        int label;
        if (second_space == string::npos)
        { // empty string
            label = stoi(str);
            str = "";
        }
        else
        {
            label = stoi(str.substr(0, second_space));
            // only populate labels vector once as it repeats
            str = str.substr(second_space + 1);
        }
        //std::cout << cr_config << ":" << label << std::endl;
        // sequence

        if (cr_config == 0)
        {
            labels.push_back(label);
        }

        if (cr_config != last_config)
        {
            sequences.push_back(vector<string>());
        }
        //if (print){
        //cout << str << endl;
        //print = false;
        //}
        //if (str.length() <= 2){
        //	cout << "Warning:|" << str << "|:Warning" << endl;
        //}
        sequences[cr_config].push_back(str);

        last_config = cr_config;
    }

    in.close();
}

void write_labels(vector<int> labels, string output)
{
    std::ofstream writer(output);

    for (int l : labels)
    {
        writer << l << ' ';
    }

    writer.close();
}

void run_sequence_miner(string train_file, string test_file, int n_ssq, string work_dir)
{
    vector<vector<string>> train_mrseqs;
    vector<int> train_labels;
    vector<vector<string>> test_mrseqs;
    vector<int> test_labels;

    read_multi_data(train_file, train_mrseqs, train_labels);
    read_multi_data(test_file, test_mrseqs, test_labels);

    //cout << train_mrseqs[0][0] << endl;

    for (int i = 0; i < train_mrseqs.size(); i++)
    {
        vector<string> mined_ssq;
        SQMiner miner('A', 100, n_ssq);
        miner.mine(train_mrseqs[i], train_labels, mined_ssq);

        if (!mined_ssq.empty())
        {
            miner.write_ft_matrix(work_dir + "/train.x." + to_string(i), train_labels);
            miner.write_mined_sequences(work_dir + "/sequences." + to_string(i));
            miner.generate_test_data(test_mrseqs[i], test_labels, work_dir + "/test.x." + to_string(i));
        }
        
    }

    write_labels(train_labels, work_dir + "/train.y");
    write_labels(test_labels, work_dir + "/test.y");
}

int main(int argc, char **argv)
{
	string train_data;
	string test_data;
	string work_dir;
	int n_subsequences = -1; //mine everything

	int opt;
	while ((opt = getopt(argc, argv, "t:T:D:n:N:w:a:s:")) != -1)
	{
		switch (opt)
		{
		case 't':
			train_data = string(optarg);
			break;
		case 'T':
			test_data = string(optarg);
			break;
		case 'D':
			work_dir = string(optarg);
			break;	
		case 's':
			n_subsequences = atoi(optarg);
			break;
		default:
			std::cout << "Usage: " << argv[0] << std::endl;
			return -1;
		}
	}
	clock_t begin = clock();
	
	run_sequence_miner(train_data, test_data, n_subsequences, work_dir);
	clock_t end = clock();
	cout << "Elapsed Time: " << double(end - begin) / CLOCKS_PER_SEC << endl;
	//tm.run(train_data);
}