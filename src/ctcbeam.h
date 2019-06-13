#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <fstream>
#include "kenlm/lm/model.hh"
#include "alphabet.h"
#include "trie.h"
using namespace lm::ngram;
const double inf = 100;

struct BeamEntry
{
    double prTotal, prNonBlank, prBlank, prText;
    bool lmApplied;
    TrieNode *lastWordNode;
    std::string lastWord;
    std::vector<std::string> completeWords;
    Model::State lmState;
};

double logSumExp(double a, double b)
{
    double mx = std::max(a, b);
    return log(exp(a - mx) + exp(b - mx)) + mx;
}

void updateEntries(std::vector<BeamEntry>& newEntries, std::vector<BeamEntry>& entries)
{
    entries.clear();
    sort(newEntries.begin(), newEntries.end(),
        [&](BeamEntry& b1, BeamEntry& b2){
            if (b1.completeWords == b2.completeWords)
                return b1.lastWord < b2.lastWord;
            return b1.completeWords < b2.completeWords;
        });
    for (BeamEntry& beam : newEntries)
    {
        if (!entries.empty() && entries.back().completeWords == beam.completeWords &&
            entries.back().lastWord == beam.lastWord)
        {
            entries.back().prTotal = logSumExp(entries.back().prTotal, beam.prTotal);
            entries.back().prNonBlank = logSumExp(entries.back().prNonBlank, beam.prNonBlank);
            entries.back().prBlank = logSumExp(entries.back().prBlank, beam.prBlank);
        }
        else
            entries.push_back(beam);
    }
}

void scoreLm(BeamEntry& beam, const Model& model)
{
    const Vocabulary &vocab = model.GetVocabulary();
    State state(model.BeginSentenceState()), out_state;
    double prob = 0;
    int order = model.Order();
    int wordCount = beam.completeWords.size();
    
    for (int i = 0; i < order - wordCount; i++)
    {
        prob = model.Score(state, vocab.Index("<s>"), out_state);
        state = out_state;
    }
    for (int i = std::max(0, wordCount - order); i < wordCount; i++)
    {
        int index = vocab.Index(beam.completeWords[i]);
        if (beam.completeWords[i].empty() || index == 0)
        {
            beam.prText = -inf;
            beam.lmApplied = true;
            return;
        }
        prob = model.Score(state, index, out_state);
        state = out_state;
    }
    if (beam.completeWords.empty())
    {
        prob = 0;
    }

    beam.prText = prob;
}

void applyLm(BeamEntry& beam, const Model& model, TrieNode *root, const Alphabet& alphabet)
{
    if (beam.lmApplied)
        return;
    beam.lmApplied = true;
    if (beam.lastWord.back() == ' ')
    {
        beam.lastWord.pop_back();
        beam.completeWords.push_back(beam.lastWord);
        beam.lastWord.clear();
        beam.lastWordNode = root;
        scoreLm(beam, model);
    }
    else
    {
        int index = alphabet.getIndex(beam.lastWord.back());
        if (beam.lastWordNode != nullptr)
            beam.lastWordNode = beam.lastWordNode->getChild(index);
        if (beam.lastWordNode == nullptr)
            beam.prText = -inf;
    }
}

void finishBeam(BeamEntry& beam, const Model& model)
{
    if (!beam.lastWord.empty())
    {
        beam.completeWords.push_back(beam.lastWord);
        beam.lastWord.clear();
        scoreLm(beam, model);
    }
}

//rows of mat are 1 longer than 'alphabet' - last element is probability of blank
std::string ctcBeamSearch(std::vector<std::vector<double>>& mat,
                          const Model& model,
                          const Alphabet& alphabet,
                          TrieNode *root,
                          int beamWidth,
                          double alpha,
                          double beta)
{
    std::function<bool(BeamEntry&, BeamEntry&)> beamCmp =
        [&](BeamEntry& b1, BeamEntry& b2) {
            return b1.prTotal + alpha * b1.prText + beta * b1.completeWords.size() >
                   b2.prTotal + alpha * b2.prText + beta * b2.completeWords.size();};
    std::vector<BeamEntry> entries;
    entries.push_back({0, -inf, 0, 0, false, root, {}, {}, model.BeginSentenceState()});
    int blankInd = alphabet.getSize();
    
    //std::cerr << blankInd << "\n";
    for (const std::vector<double>& probs : mat)
    {
        /*for (int i = 0; i <= blankInd; i++)
        {
            std::cerr << probs[i] << " ";
        }
        std::cerr << "\n";
        for (int i = 0; i < std::min(2, (int)entries.size()); i++)
        {
            for (auto word : entries[i].completeWords)
            {
                std::cerr << word << " ";
            }
            std::cerr << " | " << entries[i].lastWord << " " << entries[i].prTotal << " " << entries[i].prText << "\n";
            if (entries[i].lastWordNode != nullptr)
            {
                std::cerr << entries[i].lastWordNode->getWordCount() << "\n";
            }
            else
            {
                std::cerr << "nullptr\n";
            }
        }*/
        std::vector<BeamEntry> newEntries;
        for (BeamEntry& beam : entries)
        {
            //same labeling
            BeamEntry ent(beam);
            ent.prNonBlank = -inf;
            int index = -1;
            
            if (!beam.lastWord.empty())
                index = alphabet.getIndex(beam.lastWord.back());
            else if (!beam.completeWords.empty())
                index = alphabet.getIndex(' ');
            if (index != -1)
                ent.prNonBlank = beam.prNonBlank + log(probs[index]);

            ent.prBlank = beam.prTotal + log(probs[blankInd]);
            ent.prTotal = logSumExp(ent.prBlank, ent.prNonBlank);
            newEntries.push_back(ent);

            //different labeling
            for (int i = 0; i < blankInd; i++)
            {
                char c = alphabet.getLetter(i);
                if (beam.lastWord.empty() && c == ' ')
                    continue;
                BeamEntry ent(beam);
                ent.lastWord += c;
                if (!beam.lastWord.empty() && beam.lastWord.back() == c)
                    ent.prNonBlank = beam.prBlank + log(probs[i]);
                else
                    ent.prNonBlank = beam.prTotal + log(probs[i]);
                ent.prTotal = ent.prNonBlank;
                ent.prBlank = -inf;
                ent.lmApplied = false;
                ent.prText = 0;
                newEntries.push_back(ent);
            }
        }

        for (BeamEntry& beam : newEntries)
            applyLm(beam, model, root, alphabet);
        updateEntries(newEntries, entries);
        //std::cerr << "                             " << entries.size() << "\n";
        std::sort(entries.begin(), entries.end(), beamCmp);
        entries.resize(std::min(beamWidth, (int)entries.size()));
    }
    
    for (BeamEntry& entry : entries)
        finishBeam(entry, model);
    sort(entries.begin(), entries.end(), beamCmp);

    std::string result;
    for (std::string word : entries[0].completeWords)
        result += word + ' ';
    result.pop_back();
    //std::cerr << result << " " << entries[0].prTotal << " " << entries[0].prText << std::endl;
    return result;
}

