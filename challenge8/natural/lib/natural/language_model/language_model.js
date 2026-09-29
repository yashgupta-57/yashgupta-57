/*
Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

'use strict'

const WordTokenizer = require('../tokenizers/regexp_tokenizer').WordTokenizer

// Interpolated Kneser-Ney n-gram language model.
//
// Sentences are padded with n - 1 start symbols and one end symbol. Every n-gram of the padded
// sentences is counted. The lower-order distributions use continuation counts (the number of
// distinct left extensions of an n-gram), derived from the counts of the next higher order.
// The unigram distribution is interpolated with a uniform distribution over the vocabulary plus
// one slot for unknown words, so every word has a non-zero probability once the model is trained.

const START_SYMBOL = '<s>'
const END_SYMBOL = '</s>'
const MODES = ['word', 'character']
const DEFAULT_N = 3
const DEFAULT_DISCOUNT = 0.75

const wordTokenizer = new WordTokenizer()

// A node of a count trie. `count` is the number of times the n-gram that ends at this node was
// observed and `total` is the sum of the counts of its children, i.e. how often the sequence that
// ends here was observed as a context.
class CountNode {
  constructor () {
    this.count = 0
    this.total = 0
    this.children = new Map()
  }
}

// Trie of token sequences that all have the same length.
class CountTrie {
  constructor () {
    this.root = new CountNode()
  }

  add (ngram) {
    let parent = this.root
    let node = this.root
    for (const token of ngram) {
      parent = node
      node = parent.children.get(token)
      if (node === undefined) {
        node = new CountNode()
        parent.children.set(token, node)
      }
    }
    node.count++
    parent.total++
  }

  find (tokens) {
    let node = this.root
    for (const token of tokens) {
      node = node.children.get(token)
      if (node === undefined) {
        return undefined
      }
    }
    return node
  }

  // Yields every sequence stored in the trie
  * sequences () {
    yield * walk(this.root, [])
  }
}

function * walk (node, prefix) {
  if (node.children.size === 0) {
    if (prefix.length > 0) {
      yield prefix.slice()
    }
    return
  }
  for (const [token, child] of node.children) {
    prefix.push(token)
    yield * walk(child, prefix)
    prefix.pop()
  }
}

function normaliseOptions (options) {
  if (options === undefined || options === null) {
    options = {}
  }
  if (typeof options !== 'object') {
    throw new TypeError('options must be an object')
  }
  const n = options.n === undefined ? DEFAULT_N : options.n
  if (!Number.isInteger(n) || n < 1) {
    throw new RangeError('n must be an integer greater than or equal to 1')
  }
  const discount = options.discount === undefined ? DEFAULT_DISCOUNT : options.discount
  if (typeof discount !== 'number' || !(discount >= 0 && discount <= 1)) {
    throw new RangeError('discount must be a number between 0 and 1')
  }
  const mode = options.mode === undefined ? 'word' : options.mode
  if (!MODES.includes(mode)) {
    throw new RangeError('mode must be one of ' + MODES.join(', '))
  }
  return { n, discount, mode }
}

function copyTokens (tokens) {
  for (const token of tokens) {
    if (typeof token !== 'string') {
      throw new TypeError('tokens must be strings')
    }
  }
  return tokens.slice()
}

class LanguageModel {
  // options: { n, discount, mode }, see normaliseOptions for the defaults
  constructor (options) {
    const settings = normaliseOptions(options)
    this.n = settings.n
    this.discount = settings.discount
    this.mode = settings.mode
    this.sentences = []
    this.trained = false
    this.counts = null
    this.continuation = []
    this.vocab = new Set()
  }

  // Splits text into tokens: words for mode 'word', code points for mode 'character'
  static tokenize (text, mode) {
    if (typeof text !== 'string') {
      throw new TypeError('text must be a string')
    }
    if (mode === undefined) {
      mode = 'word'
    }
    if (mode === 'character') {
      return Array.from(text)
    }
    if (mode === 'word') {
      return wordTokenizer.tokenize(text)
    }
    throw new RangeError('mode must be one of ' + MODES.join(', '))
  }

  tokenize (text) {
    return LanguageModel.tokenize(text, this.mode)
  }

  // Turns a sentence (string or array of tokens) into a token array the model owns
  toTokens (sentence) {
    if (typeof sentence === 'string') {
      return this.tokenize(sentence)
    }
    if (Array.isArray(sentence)) {
      return copyTokens(sentence)
    }
    throw new TypeError('sentence must be a string or an array of strings')
  }

  // Adds a sentence to the training data. Returns false if the sentence has no tokens.
  // The model keeps its own copy of the tokens. Call train() to include the sentence in the counts.
  addSentence (sentence) {
    const tokens = this.toTokens(sentence)
    if (tokens.length === 0) {
      return false
    }
    this.sentences.push(tokens)
    return true
  }

  // (Re)builds the counts from all sentences added so far
  train () {
    const counts = new CountTrie()
    const vocab = new Set()
    const starts = new Array(this.n - 1).fill(START_SYMBOL)
    for (const sentence of this.sentences) {
      const padded = starts.concat(sentence, [END_SYMBOL])
      for (let i = this.n - 1; i < padded.length; i++) {
        counts.add(padded.slice(i - this.n + 1, i + 1))
      }
      for (const token of sentence) {
        vocab.add(token)
      }
      vocab.add(END_SYMBOL)
    }

    // Continuation counts: the number of distinct left extensions of each lower-order n-gram
    const continuation = []
    let higher = counts
    for (let order = this.n - 1; order >= 1; order--) {
      const trie = new CountTrie()
      for (const sequence of higher.sequences()) {
        trie.add(sequence.slice(1))
      }
      continuation[order] = trie
      higher = trie
    }

    this.counts = counts
    this.continuation = continuation
    this.vocab = vocab
    this.trained = true
  }

  // Probability of word given the preceding tokens. Only the last n - 1 tokens of the context are
  // used. A shorter context is scored with the corresponding lower-order distribution; pass
  // LanguageModel.START_SYMBOL tokens explicitly to score the beginning of a sentence.
  probability (word, context) {
    if (typeof word !== 'string') {
      throw new TypeError('word must be a string')
    }
    if (context === undefined) {
      context = []
    }
    if (!Array.isArray(context)) {
      throw new TypeError('context must be an array of strings')
    }
    for (const token of context) {
      if (typeof token !== 'string') {
        throw new TypeError('context must be an array of strings')
      }
    }
    if (!this.trained) {
      return this.uniform()
    }
    const history = context.length > this.n - 1 ? context.slice(context.length - (this.n - 1)) : context
    return this.interpolated(word, history)
  }

  uniform () {
    return 1 / (this.vocab.size + 1)
  }

  interpolated (word, history) {
    const order = history.length + 1
    const table = order === this.n ? this.counts : this.continuation[order]
    const lower = order === 1 ? this.uniform() : this.interpolated(word, history.slice(1))
    const node = table.find(history)
    if (node === undefined || node.total === 0) {
      return lower
    }
    const child = node.children.get(word)
    const count = child === undefined ? 0 : child.count
    const discount = this.discount
    return Math.max(count - discount, 0) / node.total +
      (discount * node.children.size / node.total) * lower
  }

  // Natural logarithm of the probability of a sentence including its end symbol
  logProbability (sentence) {
    const tokens = this.toTokens(sentence)
    const padded = new Array(this.n - 1).fill(START_SYMBOL).concat(tokens, [END_SYMBOL])
    let logProbability = 0
    for (let i = this.n - 1; i < padded.length; i++) {
      logProbability += Math.log(this.probability(padded[i], padded.slice(i - this.n + 1, i)))
    }
    return logProbability
  }

  // Sorted list of the tokens seen during training (including the end symbol)
  vocabulary () {
    return Array.from(this.vocab).sort()
  }

  toJSON () {
    return {
      n: this.n,
      discount: this.discount,
      mode: this.mode,
      trained: this.trained,
      sentences: this.sentences.map(sentence => sentence.slice())
    }
  }

  // Builds an independent model from the output of toJSON(). The model is retrained from its
  // sentences if the original model had been trained.
  static restore (json) {
    if (json === null || typeof json !== 'object') {
      throw new TypeError('json must be an object created by LanguageModel#toJSON')
    }
    if (!Array.isArray(json.sentences)) {
      throw new TypeError('json.sentences must be an array of token arrays')
    }
    const model = new LanguageModel({ n: json.n, discount: json.discount, mode: json.mode })
    for (const sentence of json.sentences) {
      if (!Array.isArray(sentence)) {
        throw new TypeError('json.sentences must be an array of token arrays')
      }
      model.addSentence(sentence)
    }
    if (json.trained) {
      model.train()
    }
    return model
  }
}

LanguageModel.START_SYMBOL = START_SYMBOL
LanguageModel.END_SYMBOL = END_SYMBOL

module.exports = LanguageModel
