#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <fstream>
const std::string globalClasses = "abcdefghijklmnopqrstuvwxyz \'";
const double LMFACTOR = 0.01;

class BeamEntry
{
public:
    double prTotal, prNonBlank, prBlank, prText;
    bool lmApplied;
    std::vector<int> labeling;
    
    BeamEntry()
      : prTotal(0), prNonBlank(0), prBlank(0), prText(1), lmApplied(false), labeling({}) {}
    
    BeamEntry(double _prTotal, double _prNonBlank, double _prBlank,
              double _prText, bool _lmApplied, std::vector<int> _labeling)
      : prTotal(_prTotal), prNonBlank(_prNonBlank), prBlank(_prBlank),
        prText(_prBlank), lmApplied(_lmApplied), labeling(_labeling) {}
    
    bool operator <(BeamEntry& oth)
    {
        return prTotal * prText > oth.prTotal * oth.prText;
    }
};

class LM
{
public:
    std::map<std::string, long long> probs;
    std::map<int, long long> sizes;
    int length;
    
    LM() : probs(), sizes(), length(0) {}
    LM(int len) : probs(), sizes(), length(len) {}
    
    double getProb(std::string ngram) const
    {
        int len = ngram.size();
        if (probs.count(ngram) && sizes.count(len) && sizes.at(len) != 0)
        {
            return (double)probs.at(ngram) / sizes.at(len);
        }
        return 0;
    }
};
const LM emptyLM(0);

LM languageModelFromFile(const char *file)
{
    std::ifstream ngramFile(file);

    if (!ngramFile.is_open())
    {
        std::cerr << "Cannot open ngram file, returning empty language model\n";
        return LM(0);
    }
    
    LM res;
    std::string ngram;
    long long count;
    while (ngramFile >> ngram >> count)
    {
        int length = ngram.size();
        res.sizes[length] += count;
        res.probs[ngram] = count;
        res.length = std::max(res.length, length);
    }
    return res;
}

void updateEntries(std::vector<BeamEntry>& newEntries, std::vector<BeamEntry>& entries)
{
    entries.clear();
    sort(newEntries.begin(), newEntries.end(),
         [&](BeamEntry& b1, BeamEntry& b2){return b1.labeling < b2.labeling;});
    for (BeamEntry& beam : newEntries)
    {
        if (!entries.empty() && entries.back().labeling == beam.labeling)
        {
            entries.back().prTotal += beam.prTotal;
            entries.back().prNonBlank += beam.prNonBlank;
            entries.back().prBlank += beam.prBlank;
        }
        else
            entries.push_back(beam);
    }
}

std::string getNgram(BeamEntry& beam, std::string& classes, int length)
{
    std::string res;
    for (int i = (int)beam.labeling.size() - 1; i >= 0; i--)
    {
        int ind = beam.labeling[i];
        if (classes[ind] >= 'a' && classes[ind] <= 'z')
            res += classes[ind];
        if ((int)res.size() >= length)
            break;
    }
    reverse(res.begin(), res.end());
	return res;
}

void applyLm(BeamEntry& beam, std::string& classes, double lmFactor, const LM& lm)
{
    if (beam.lmApplied)
        return;
   	std::string ngram = getNgram(beam, classes, lm.length);
    beam.prText *= pow(lm.getProb(ngram), lmFactor);
    beam.lmApplied = true;
}

//rows of mat are 1 longer than 'classes' - last element is probability of blank
std::string ctcBeamSearch(std::vector<std::vector<double>>& mat,
                          int beamWidth = 15,
                          std::string classes = globalClasses,
                          const LM &languageModel = emptyLM)
{
    std::vector<BeamEntry> entries;
    entries.push_back(BeamEntry(1, 0, 1, 1, false, {}));
    int blankInd = classes.size();
    
    for (const std::vector<double>& probs : mat)
    {
        std::vector<BeamEntry> newEntries;
        for (BeamEntry& beam : entries)
        {
            //same labeling
            BeamEntry ent(beam);
            ent.prNonBlank = 0;
            if (!beam.labeling.empty())
                ent.prNonBlank = beam.prNonBlank * probs[beam.labeling.back()];
            
            ent.prBlank = beam.prTotal * probs[blankInd];
            ent.prTotal = ent.prBlank + ent.prNonBlank;
            newEntries.push_back(ent);

            //different labeling
            for (int i = 0; i < blankInd; i++)
            {
                BeamEntry ent(beam);
                ent.labeling.push_back(i);
                if (!beam.labeling.empty() && beam.labeling.back() == i)
                    ent.prNonBlank = beam.prBlank * probs[i];
                else
                    ent.prNonBlank = beam.prTotal * probs[i];
                ent.prTotal = ent.prNonBlank;
                ent.prBlank = 0;
                ent.lmApplied = false;
                ent.prText = 1;
                newEntries.push_back(ent);
            }
        }

        updateEntries(newEntries, entries);
        for (BeamEntry& beam : entries)
            applyLm(beam, classes, LMFACTOR, languageModel);
        std::sort(entries.begin(), entries.end());
        entries.resize(std::min(beamWidth, (int)entries.size()));
    }
    
    std::string result;
    for (int index : entries[0].labeling)
        result += classes[index];
    return result;
}

//examples of usage
/*int main()
{
    std::string classes = "ab";
    std::vector<std::vector<double>> mat = {{0.4, 0, 0.6}, {0.4, 0, 0.6}};
    std::cerr << "Test beam search\n";
    std::string expected = "a";
    std::string actual = ctcBeamSearch(mat, classes);
    std::cerr << "Expected: \"" << expected << "\"\n";
    std::cerr << "Actual: \"" << actual << "\"\n";
    if (expected == actual)
        std::cerr << "OK\n";
    else
        std::cerr << "ERROR\n";
    int n, m;
    std::cin >> n >> m;
    std::vector<std::vector<double>> X;
    for (int i = 0; i < n; i++)
    {
    	X.push_back(std::vector<double>{});
    	for (int j = 0; j < m; j++)
    	{
    		double x;
    		std::cin >> x;
    		X[i].push_back(x);
    	}
    }
    std::cerr << X.size() << "\n";
    std::cout << ctcBeamSearch(X, 15) << "\n";
}*/
