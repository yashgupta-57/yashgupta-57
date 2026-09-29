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

const natural = require('../lib/natural')
const LanguageModel = natural.LanguageModel

const START = LanguageModel.START_SYMBOL
const END = LanguageModel.END_SYMBOL

// Sums the probability of every known word plus one unknown word: an interpolated model must
// distribute exactly one unit of probability mass over the vocabulary and the unknown slot.
function totalMass (model, context) {
  let sum = 0
  model.vocabulary().forEach(function (word) {
    sum += model.probability(word, context)
  })
  return sum + model.probability('a-word-that-was-never-seen', context)
}

describe('LanguageModel', function () {
  it('is exported from the root of the library', function () {
    expect(typeof natural.LanguageModel).toBe('function')
    expect(natural.LanguageModel).toBe(require('../lib/natural/language_model').LanguageModel)
  })

  describe('constructor', function () {
    it('uses a trigram model with discount 0.75 in word mode by default', function () {
      const model = new LanguageModel()
      expect(model.n).toBe(3)
      expect(model.discount).toBe(0.75)
      expect(model.mode).toBe('word')
      expect(model.trained).toBe(false)
      expect(model.sentences).toEqual([])
    })

    it('accepts n, discount and mode options', function () {
      const model = new LanguageModel({ n: 2, discount: 0.5, mode: 'character' })
      expect(model.n).toBe(2)
      expect(model.discount).toBe(0.5)
      expect(model.mode).toBe('character')
    })

    it('rejects invalid options', function () {
      expect(() => new LanguageModel({ n: 0 })).toThrowError(RangeError)
      expect(() => new LanguageModel({ n: 1.5 })).toThrowError(RangeError)
      expect(() => new LanguageModel({ n: '3' })).toThrowError(RangeError)
      expect(() => new LanguageModel({ discount: -0.1 })).toThrowError(RangeError)
      expect(() => new LanguageModel({ discount: 1.1 })).toThrowError(RangeError)
      expect(() => new LanguageModel({ discount: NaN })).toThrowError(RangeError)
      expect(() => new LanguageModel({ mode: 'syllable' })).toThrowError(RangeError)
      expect(() => new LanguageModel('word')).toThrowError(TypeError)
    })

    it('exposes the padding symbols', function () {
      expect(START).toBe('<s>')
      expect(END).toBe('</s>')
    })
  })

  describe('tokenize', function () {
    it('splits words in word mode', function () {
      expect(LanguageModel.tokenize('The cat, the dog!')).toEqual(['The', 'cat', 'the', 'dog'])
      expect(new LanguageModel().tokenize('a b')).toEqual(['a', 'b'])
    })

    it('splits code points in character mode', function () {
      expect(LanguageModel.tokenize('ab c', 'character')).toEqual(['a', 'b', ' ', 'c'])
      expect(new LanguageModel({ mode: 'character' }).tokenize('😀!')).toEqual(['😀', '!'])
    })

    it('rejects non-strings and unknown modes', function () {
      expect(() => LanguageModel.tokenize(42)).toThrowError(TypeError)
      expect(() => LanguageModel.tokenize('a', 'syllable')).toThrowError(RangeError)
    })
  })

  describe('addSentence', function () {
    it('tokenises strings according to the mode', function () {
      const words = new LanguageModel()
      expect(words.addSentence('I love cats')).toBe(true)
      expect(words.sentences).toEqual([['I', 'love', 'cats']])

      const characters = new LanguageModel({ mode: 'character' })
      characters.addSentence('ab')
      expect(characters.sentences).toEqual([['a', 'b']])
    })

    it('ignores sentences without tokens', function () {
      const model = new LanguageModel()
      expect(model.addSentence('')).toBe(false)
      expect(model.addSentence('   ,,, ')).toBe(false)
      expect(model.addSentence([])).toBe(false)
      expect(model.sentences).toEqual([])
    })

    it('keeps its own copy of a token array', function () {
      const model = new LanguageModel({ n: 2 })
      const tokens = ['a', 'b']
      model.addSentence(tokens)
      tokens[0] = 'changed'
      tokens.push('extra')
      expect(model.sentences).toEqual([['a', 'b']])
      model.train()
      expect(model.probability('b', ['a'])).toBeGreaterThan(model.probability('b', ['changed']))
    })

    it('rejects sentences that are neither strings nor arrays of strings', function () {
      const model = new LanguageModel()
      expect(() => model.addSentence(42)).toThrowError(TypeError)
      expect(() => model.addSentence(['a', 1])).toThrowError(TypeError)
      expect(() => model.addSentence(null)).toThrowError(TypeError)
    })
  })

  describe('probability before training', function () {
    it('is the same for every word and context', function () {
      const model = new LanguageModel()
      model.addSentence('some words that were added but not trained')
      expect(model.probability('some')).toBe(1)
      expect(model.probability('other', ['some', 'words'])).toBe(1)
      expect(model.vocabulary()).toEqual([])
    })
  })

  describe('bigram model', function () {
    // Padded sentences: <s> a b </s> and <s> a c </s>
    // Bigram counts: (<s> a) 2, (a b) 1, (b </s>) 1, (a c) 1, (c </s>) 1
    // Continuation counts: a 1, b 1, c 1, </s> 2 (total 5, 4 types); vocabulary size 4
    let model
    beforeEach(function () {
      model = new LanguageModel({ n: 2 })
      model.addSentence(['a', 'b'])
      model.addSentence(['a', 'c'])
      model.train()
    })

    it('is trained and knows its vocabulary', function () {
      expect(model.trained).toBe(true)
      expect(model.vocabulary()).toEqual(['</s>', 'a', 'b', 'c'])
    })

    it('interpolates the discounted count with the continuation probability', function () {
      // P(b) = (1 - 0.75) / 5 + 0.75 * 4 / 5 * 1 / 5 = 0.17
      expect(model.probability('b')).toBeCloseTo(0.17, 12)
      // P(</s>) = (2 - 0.75) / 5 + 0.12 = 0.37
      expect(model.probability(END)).toBeCloseTo(0.37, 12)
      // P(b | a) = (1 - 0.75) / 2 + 0.75 * 2 / 2 * 0.17 = 0.2525
      expect(model.probability('b', ['a'])).toBeCloseTo(0.2525, 12)
      expect(model.probability('c', ['a'])).toBeCloseTo(0.2525, 12)
      // P(a | <s>) = (2 - 0.75) / 2 + 0.75 * 1 / 2 * 0.17 = 0.68875
      expect(model.probability('a', [START])).toBeCloseTo(0.68875, 12)
      // P(</s> | b) = (1 - 0.75) / 1 + 0.75 * 1 / 1 * 0.37 = 0.5275
      expect(model.probability(END, ['b'])).toBeCloseTo(0.5275, 12)
    })

    it('reserves probability mass for unseen words', function () {
      // P(unknown) = 0.75 * 4 / 5 * 1 / 5 = 0.12 and P(unknown | a) = 0.75 * 0.12
      expect(model.probability('zzz')).toBeCloseTo(0.12, 12)
      expect(model.probability('zzz', ['a'])).toBeCloseTo(0.09, 12)
      // P(</s> | a) backs off entirely: 0.75 * 0.37
      expect(model.probability(END, ['a'])).toBeCloseTo(0.2775, 12)
    })

    it('backs off to the lower order for an unseen context', function () {
      expect(model.probability('b', ['zzz'])).toBeCloseTo(model.probability('b'), 12)
      expect(model.probability('b', [END])).toBeCloseTo(model.probability('b'), 12)
    })

    it('distributes exactly one unit of probability mass per context', function () {
      expect(totalMass(model, [])).toBeCloseTo(1, 12)
      expect(totalMass(model, ['a'])).toBeCloseTo(1, 12)
      expect(totalMass(model, [START])).toBeCloseTo(1, 12)
      expect(totalMass(model, ['zzz'])).toBeCloseTo(1, 12)
    })

    it('scores whole sentences with their end symbol', function () {
      const expected = Math.log(0.68875) + Math.log(0.2525) + Math.log(0.5275)
      expect(model.logProbability(['a', 'b'])).toBeCloseTo(expected, 12)
      expect(model.logProbability('a b')).toBeCloseTo(expected, 12)
      expect(model.logProbability([])).toBeCloseTo(Math.log(model.probability(END, [START])), 12)
      expect(model.logProbability(['a', 'b'])).toBeGreaterThan(model.logProbability(['b', 'a']))
    })

    it('keeps the trained probabilities until it is trained again', function () {
      const before = model.probability('c', ['a'])
      model.addSentence(['a', 'c'])
      expect(model.probability('c', ['a'])).toBe(before)
      model.train()
      expect(model.probability('c', ['a'])).toBeGreaterThan(before)
    })
  })

  describe('trigram model', function () {
    // Padded sentences: <s> <s> a b c </s> and <s> <s> a b d </s>
    let model
    beforeEach(function () {
      model = new LanguageModel()
      model.addSentence(['a', 'b', 'c'])
      model.addSentence(['a', 'b', 'd'])
      model.train()
    })

    it('computes continuation counts for every lower order', function () {
      // Unigram continuation counts: a 1, b 1, c 1, d 1, </s> 2 (total 6, 5 types), vocabulary size 5
      // P(c) = (1 - 0.75) / 6 + 0.75 * 5 / 6 * 1 / 6 = 0.1458333...
      expect(model.probability('c')).toBeCloseTo(0.25 / 6 + 0.625 / 6, 12)
      // P(c | b) = (1 - 0.75) / 2 + 0.75 * 2 / 2 * P(c) = 0.234375
      expect(model.probability('c', ['b'])).toBeCloseTo(0.234375, 12)
      // P(c | a b) = (1 - 0.75) / 2 + 0.75 * 2 / 2 * P(c | b) = 0.30078125
      expect(model.probability('c', ['a', 'b'])).toBeCloseTo(0.30078125, 12)
      expect(model.probability('d', ['a', 'b'])).toBeCloseTo(0.30078125, 12)
      // P(</s> | a b) = 0.75 * P(</s> | b) = 0.75 * 0.75 * P(</s>) = 0.75 * 0.75 * 0.3125
      expect(model.probability(END, ['a', 'b'])).toBeCloseTo(0.17578125, 12)
    })

    it('only uses the last n - 1 tokens of the context', function () {
      expect(model.probability('c', ['x', 'y', 'a', 'b'])).toBe(model.probability('c', ['a', 'b']))
    })

    it('distributes exactly one unit of probability mass per context', function () {
      expect(totalMass(model, ['a', 'b'])).toBeCloseTo(1, 12)
      expect(totalMass(model, ['q', 'b'])).toBeCloseTo(1, 12)
      expect(totalMass(model, ['b'])).toBeCloseTo(1, 12)
      expect(totalMass(model, [START, START])).toBeCloseTo(1, 12)
    })

    it('validates its arguments', function () {
      expect(() => model.probability(1)).toThrowError(TypeError)
      expect(() => model.probability('a', 'b')).toThrowError(TypeError)
      expect(() => model.probability('a', [1])).toThrowError(TypeError)
      expect(() => model.logProbability(42)).toThrowError(TypeError)
    })
  })

  describe('other configurations', function () {
    it('supports unigram models', function () {
      const model = new LanguageModel({ n: 1 })
      model.addSentence(['a', 'a', 'b'])
      model.train()
      // counts: a 2, b 1, </s> 1 (total 4, 3 types); vocabulary size 3
      expect(model.probability('a')).toBeCloseTo((2 - 0.75) / 4 + 0.75 * 3 / 4 / 4, 12)
      expect(model.probability('a', ['b'])).toBe(model.probability('a'))
      expect(totalMass(model, [])).toBeCloseTo(1, 12)
    })

    it('reduces to maximum likelihood estimation with discount 0', function () {
      const model = new LanguageModel({ n: 2, discount: 0 })
      model.addSentence(['a', 'b'])
      model.addSentence(['a', 'b'])
      model.addSentence(['a', 'c'])
      model.train()
      expect(model.probability('b', ['a'])).toBeCloseTo(2 / 3, 12)
      expect(model.probability('c', ['a'])).toBeCloseTo(1 / 3, 12)
      expect(model.probability('zzz', ['a'])).toBe(0)
    })

    it('supports character models', function () {
      const model = new LanguageModel({ n: 2, mode: 'character' })
      model.addSentence('ab')
      model.addSentence('ab')
      model.addSentence('ac')
      model.train()
      expect(model.vocabulary()).toEqual(['</s>', 'a', 'b', 'c'])
      expect(model.probability('b', ['a'])).toBeGreaterThan(model.probability('c', ['a']))
      expect(model.logProbability('ab')).toBeGreaterThan(model.logProbability('ac'))
    })
  })

  describe('toJSON and restore', function () {
    let model
    beforeEach(function () {
      model = new LanguageModel({ n: 2, discount: 0.5 })
      model.addSentence(['a', 'b'])
      model.addSentence('a c')
      model.train()
    })

    it('serialises the options, the training state and copies of the sentences', function () {
      const json = model.toJSON()
      expect(json).toEqual({ n: 2, discount: 0.5, mode: 'word', trained: true, sentences: [['a', 'b'], ['a', 'c']] })
      expect(JSON.parse(JSON.stringify(model))).toEqual(json)
      json.sentences[0][0] = 'changed'
      json.sentences.push(['x'])
      expect(model.sentences).toEqual([['a', 'b'], ['a', 'c']])
    })

    it('restores an equivalent trained model', function () {
      const copy = LanguageModel.restore(JSON.parse(JSON.stringify(model)))
      expect(copy).not.toBe(model)
      expect(copy.n).toBe(2)
      expect(copy.discount).toBe(0.5)
      expect(copy.trained).toBe(true)
      expect(copy.sentences).toEqual(model.sentences)
      expect(copy.vocabulary()).toEqual(model.vocabulary())
      expect(copy.probability('b', ['a'])).toBe(model.probability('b', ['a']))
      expect(copy.logProbability(['a', 'c'])).toBe(model.logProbability(['a', 'c']))
    })

    it('restores an untrained model as untrained', function () {
      const untrained = new LanguageModel({ mode: 'character' })
      untrained.addSentence('ab')
      const copy = LanguageModel.restore(untrained.toJSON())
      expect(copy.trained).toBe(false)
      expect(copy.mode).toBe('character')
      expect(copy.sentences).toEqual([['a', 'b']])
      expect(copy.probability('a')).toBe(1)
      copy.train()
      expect(copy.probability('b', ['a'])).toBeGreaterThan(copy.probability('c', ['a']))
    })

    it('is independent of the JSON object it was restored from', function () {
      const state = model.toJSON()
      const copy = LanguageModel.restore(state)
      state.sentences[0][0] = 'changed'
      state.sentences[1].push('extra')
      state.sentences.push(['q'])
      expect(copy.sentences).toEqual([['a', 'b'], ['a', 'c']])
      copy.train()
      expect(copy.probability('b', ['a'])).toBe(model.probability('b', ['a']))
      expect(copy.vocabulary()).toEqual(['</s>', 'a', 'b', 'c'])
    })

    it('does not share sentences with the original model', function () {
      const copy = LanguageModel.restore(model.toJSON())
      copy.addSentence(['z'])
      expect(model.sentences.length).toBe(2)
      expect(copy.sentences.length).toBe(3)
    })

    it('rejects malformed input', function () {
      expect(() => LanguageModel.restore(null)).toThrowError(TypeError)
      expect(() => LanguageModel.restore('{}')).toThrowError(TypeError)
      expect(() => LanguageModel.restore({ n: 2 })).toThrowError(TypeError)
      expect(() => LanguageModel.restore({ n: 2, sentences: ['a b'] })).toThrowError(TypeError)
      expect(() => LanguageModel.restore({ n: 0, sentences: [] })).toThrowError(RangeError)
    })
  })
})
