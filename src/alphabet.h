#ifndef ALPHABET_H
#define ALPHABET_H

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <fstream>
#include <limits>
#include "kenlm/lm/model.hh"
using namespace lm::ngram;

class Alphabet
{
public:
    Alphabet(const char *alphabet)
    {
        len = strlen(alphabet);
        intToChar.resize(len);
        for (int i = 0; i < len; i++)
        {
            maxChar = std::max(maxChar, (int)alphabet[i]);
        }
        charToInt.resize(maxChar + 1);
        
        for (int i = 0; i < len; i++)
        {
            intToChar[i] = alphabet[i];
            charToInt[alphabet[i]] = i;
        }
    }
    
    std::vector<int> stringToClasses(const std::string& word) const
    {
        std::vector<int> result;
        for (char c : word)
        {
            result.push_back(charToInt[c]);
        }
        return result;
    }
    
    std::string classesToString(const std::vector<int>& classes) const
    {
        std::string result;
        result.reserve(classes.size());
        for (int i : classes)
        {
            result += intToChar[i];
        }
        return result;
    }
    
    char getLetter(int ind) const
    {
    	return intToChar[ind];
    }
    
    int getIndex(char c) const
    {
    	return charToInt[c];
    }
    
    int getSize() const
    {
        return len;
    }
private:
    int len, maxChar;
    std::vector<char> intToChar;
    std::vector<int> charToInt;
};


#endif /* ALPHABET_H */
