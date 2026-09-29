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
// Tokens seen fewer than minCount times in the training data are replaced by the unknown symbol.
// Sentences are padded with n - 1 start symbols and one end symbol. Every n-gram of the padded
// sentences is counted. The lower-order distributions use continuation counts (the number of
// distinct left extensions of an n-gram), derived from the counts of the next higher order.
// The unigram distribution is interpolated with a uniform distribution over the vocabulary plus
// the unknown symbol, so every word has a non-zero probability once the model is trained. Words
// that are not in the vocabulary are scored as the unknown symbol.

const START_SYMBOL = '<s>'
const END_SYMBOL = '</s>'
const UNKNOWN_SYMBOL = '<unk>'
const UNITS = ['word', 'char']
const DEFAULT_N = 3
const DEFAULT_DISCOUNT = 0.75
const DEFAULT_MIN_COUNT = 1

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
  const unit = options.unit === undefined ? 'word' : options.unit
  if (!UNITS.includes(unit)) {
    throw new RangeError('unit must be one of ' + UNITS.join(', '))
  }
  const minCount = options.minCount === undefined ? DEFAULT_MIN_COUNT : options.minCount
  if (!Number.isInteger(minCount) || minCount < 1) {
    throw new RangeError('minCount must be an integer greater than or equal to 1')
  }
  return { n, discount, unit, minCount }
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
  // options: { n, discount, unit, minCount }, see normaliseOptions for the defaults
  constructor (options) {
    const settings = normaliseOptions(options)
    this.n = settings.n
    this.discount = settings.discount
    this.unit = settings.unit
    this.minCount = settings.minCount
    this.sentences = []
    this.trained = false
    this.counts = null
    this.continuation = []
    this.vocab = new Set()
  }

  // Splits text into tokens: words for unit 'word', code points for unit 'char'
  static tokenize (text, unit) {
    if (typeof text !== 'string') {
      throw new TypeError('text must be a string')
    }
    if (unit === undefined) {
      unit = 'word'
    }
    if (unit === 'char') {
      return Array.from(text)
    }
    if (unit === 'word') {
      return wordTokenizer.tokenize(text)
    }
    throw new RangeError('unit must be one of ' + UNITS.join(', '))
  }

  tokenize (text) {
    return LanguageModel.tokenize(text, this.unit)
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
    // The vocabulary: tokens seen at least minCount times, plus the end symbol
    const frequencies = new Map()
    for (const sentence of this.sentences) {
      for (const token of sentence) {
        frequencies.set(token, (frequencies.get(token) || 0) + 1)
      }
    }
    const vocab = new Set()
    for (const [token, frequency] of frequencies) {
      if (frequency >= this.minCount) {
        vocab.add(token)
      }
    }
    if (this.sentences.length > 0) {
      vocab.add(END_SYMBOL)
    }
    const known = token => (vocab.has(token) ? token : UNKNOWN_SYMBOL)

    const counts = new CountTrie()
    const starts = new Array(this.n - 1).fill(START_SYMBOL)
    for (const sentence of this.sentences) {
      const padded = starts.concat(sentence.map(known), [END_SYMBOL])
      for (let i = this.n - 1; i < padded.length; i++) {
        counts.add(padded.slice(i - this.n + 1, i + 1))
      }
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

  // Maps a token that is not in the vocabulary to the unknown symbol; padding symbols pass through
  normaliseToken (token) {
    if (token === START_SYMBOL || token === END_SYMBOL || this.vocab.has(token)) {
      return token
    }
    return UNKNOWN_SYMBOL
  }

  // Probability of word given the preceding tokens. Only the last n - 1 tokens of the context are
  // used. A shorter context is scored with the corresponding lower-order distribution; pass
  // LanguageModel.START_SYMBOL tokens explicitly to score the beginning of a sentence. Words and
  // context tokens outside the vocabulary are scored as the unknown symbol.
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
    return this.interpolated(this.normaliseToken(word), history.map(token => this.normaliseToken(token)))
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

  // Sorted list of the tokens kept during training (including the end symbol)
  vocabulary () {
    return Array.from(this.vocab).sort()
  }

  toJSON () {
    return {
      n: this.n,
      discount: this.discount,
      unit: this.unit,
      minCount: this.minCount,
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
    const model = new LanguageModel({ n: json.n, discount: json.discount, unit: json.unit, minCount: json.minCount })
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
LanguageModel.UNKNOWN_SYMBOL = UNKNOWN_SYMBOL

module.exports = LanguageModel
