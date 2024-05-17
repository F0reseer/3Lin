# PPM feature

Regular text predicting transformer takes embedding of current character/token as input. For absolute position encoding embedding of position is added to the input. This information about position helps transformer to predict next token better. With this mechanism we can provide any additional useful information to the transformer - just add embeddings of additional features which might be useful for the transformer task - predicting text.

Text compression can be cast as text prediction task. So we can try to leverage some text compression techniques to create useful features for transformer. For example, we can take one of the PPM family algorithms and feed features from it to transformer along with character & position embeddings.

One of the strongest features for text compression is longest match continuation. [LZ77](https://en.wikipedia.org/wiki/LZ77_and_LZ78) algorithm is based on this idea. To compute it we search for the exact match in history - same fragment of text as the last N characters. Our prediction is that the next character will be the same as the one followed found exact match. Among multiple matches the longest match performs best. So to compute this feature we have to scan history, find the longest exact match and feed embedding of it's continuation to transformer.

To experiment with this technique set USE_PPM to true in train script.

