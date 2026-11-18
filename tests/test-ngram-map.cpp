#include "ngram-map.h"

#include <cstdio>
#include <cstring>

static llama_tokens make_prompt(bool majority_first) {
    llama_tokens prompt{999};
    const int values[] = {majority_first ? 200 : 300, 200, 200, majority_first ? 300 : 200};
    for (int value : values) {
        for (int i = 0; i < 12; ++i) {
            prompt.push_back(100 + i);
        }
        for (int i = 0; i < 48; ++i) {
            prompt.push_back(value + i);
        }
        prompt.push_back(999);
    }
    for (int i = 0; i < 11; ++i) {
        prompt.push_back(100 + i);
    }
    return prompt;
}

static void test_selection(bool majority_first, bool hashed, bool key_only) {
    const auto prompt = make_prompt(majority_first);
    common_ngram_map map(12, 48, key_only, 1);
    if (!hashed) {
        map.key_map.clear();
    }
    common_ngram_map_begin(map, prompt);
    llama_tokens draft;
    common_ngram_map_draft(map, prompt, 111, draft);
    const int expected = key_only && majority_first ? 300 : 200;
    GGML_ASSERT(draft.size() == 48);
    for (int i = 0; i < 48; ++i) {
        GGML_ASSERT(draft[i] == expected + i);
    }
    if (!key_only) {
        const auto & key = map.keys[map.last_draft_key_idx];
        GGML_ASSERT(prompt[key.values[map.last_draft_value_idx].value_idx] == expected);
    }
}

static void test_repeated_lookup(bool hashed) {
    const auto prompt = make_prompt(true);
    common_ngram_map map(12, 48, false, 1);
    if (!hashed) {
        map.key_map.clear();
    }
    common_ngram_map_begin(map, prompt);
    llama_tokens draft;
    common_ngram_map_draft(map, prompt, 111, draft);
    GGML_ASSERT(map.last_draft_created);
    const auto before = map.keys[map.last_draft_key_idx];
    common_ngram_map_accept(map, 5);
    draft.clear();
    common_ngram_map_draft(map, prompt, 111, draft);
    const auto & after = map.keys[0];
    for (int i = 0; i < COMMON_NGRAM_MAX_VALUES; ++i) {
        GGML_ASSERT(before.values[i].value_num == after.values[i].value_num);
    }
    GGML_ASSERT(draft.size() == 5);
    GGML_ASSERT(draft.front() == 200);
}

int main(int argc, char ** argv) {
    const bool selection = argc == 1 || std::strcmp(argv[1], "selection") == 0;
    const bool repeated = argc == 1 || std::strcmp(argv[1], "repeated") == 0;
    GGML_ASSERT(selection || repeated);
    for (bool hashed : {false, true}) {
        if (selection) {
            for (bool majority_first : {false, true}) {
                test_selection(majority_first, hashed, false);
                test_selection(majority_first, hashed, true);
            }
        }
        if (repeated) {
            test_repeated_lookup(hashed);
        }
    }
    std::puts("ngram map tests passed");
    return 0;
}
