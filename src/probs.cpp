/*
helper for trie creation - give name of binary language model as argument and vocabulary to stdin,
get vocabulary with probabilities to stdout
*/
#include "lm/model.hh"
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char** argv)
{
    using namespace lm::ngram;
    if (argc < 1)
    {
        cerr << "Give binary lm file.\n";
        return 0;
    }
    Model model(argv[1]);
    State state(model.BeginSentenceState()), out_state;
    const Vocabulary &vocab = model.GetVocabulary();
    std::string word;
    while (cin >> word)
    {
        double prob = model.Score(state, vocab.Index(word), out_state);
        cout << word << " " << prob << "\n";
    }
}
