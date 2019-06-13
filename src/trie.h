#ifndef TRIE_H
#define TRIE_H

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <fstream>
#include <limits>
#include "kenlm/lm/model.hh"
#include "alphabet.h"
using namespace lm::ngram;

class TrieNode
{
public:
    TrieNode(int size) : minScore(std::numeric_limits<double>::max()),
                         wordCount(0),
                         alphaSize(size)
    {
        children = new TrieNode*[size]();
    }
    
    void insert(std::vector<int> word, double score)
    {
        insert_(word, 0, score);
    }
    
    ~TrieNode()
    {
        for (int i = 0; i < alphaSize; i++)
        {
            delete children[i];
        }
        delete children;
    }
    
    TrieNode *getChild(int index) const
    {
    	if (index < 0 || index >= alphaSize)
    		return nullptr;
    	return children[index];
    }
    
    int getWordCount() const
    {
    	return wordCount;
    }
private:
    double minScore;
    int wordCount, alphaSize;
    TrieNode **children;
    
    void insert_(const std::vector<int>& word, unsigned pos, double score)
    {
        wordCount++;
        minScore = std::min(minScore, score);
        if (word.size() == pos)
        {
            return;
        }
        if (children[word[pos]] == nullptr)
        {
            children[word[pos]] = new TrieNode(alphaSize);
        }
        children[word[pos]]->insert_(word, pos + 1, score);
    }
};

TrieNode *trieFromFile(const char *file, const Alphabet& alphabet)
{
    std::ifstream f(file);
    std::string word;
    double score;
    TrieNode *root = new TrieNode(alphabet.getSize());
    while (f >> word >> score)
    {
        std::vector<int> classes = alphabet.stringToClasses(word);
        root->insert(classes, score);
    }
    return root;
}

#endif /* TRIE_H */
