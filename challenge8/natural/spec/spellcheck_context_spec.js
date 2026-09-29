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
const Spellcheck = natural.Spellcheck
const LanguageModel = natural.LanguageModel

describe('Spellcheck with a language model', function () {
  const wordlist = ['the', 'the', 'the', 'ten', 'tea', 'tee']

  // Bigram model: 'tea' follows 'drink', 'ten' follows 'count'; 'the' and 'tee' are unknown
  function trainedModel () {
    const model = new LanguageModel({ n: 2 })
    model.addSentence(['drink', 'tea'])
    model.addSentence(['drink', 'tea'])
    model.addSentence(['count', 'ten'])
    model.train()
    return model
  }

  describe('constructor', function () {
    it('works without a language model', function () {
      expect(new Spellcheck(['cat']).languageModel).toBeNull()
      expect(new Spellcheck(['cat'], null).languageModel).toBeNull()
      expect(new Spellcheck(['cat'], undefined).languageModel).toBeNull()
    })

    it('keeps the language model', function () {
      const model = new LanguageModel()
      expect(new Spellcheck(['cat'], model).languageModel).toBe(model)
    })

    it('accepts any object with a probability method', function () {
      const scorer = { probability: function () { return 1 } }
      expect(new Spellcheck(['cat'], scorer).languageModel).toBe(scorer)
    })

    it('rejects objects without a probability method', function () {
      expect(() => new Spellcheck(['cat'], {})).toThrowError(TypeError)
      expect(() => new Spellcheck(['cat'], 'model')).toThrowError(TypeError)
    })
  })

  describe('wordlist frequencies', function () {
    it('counts words named like Object.prototype members', function () {
      const spellcheck = new Spellcheck(['constructor', 'constructor', 'constructoq', 'toString'])
      expect(spellcheck.word2frequency.constructor).toBe(2)
      expect(spellcheck.word2frequency.constructoq).toBe(1)
      expect(spellcheck.word2frequency.toString).toBe(1)
      expect(spellcheck.word2frequency.valueOf).toBeUndefined()
      expect(spellcheck.isCorrect('constructor')).toBe(true)
      expect(spellcheck.isCorrect('valueOf')).toBe(false)
    })

    it('rank such words by frequency without a language model', function () {
      const spellcheck = new Spellcheck(['constructor', 'constructor', 'constructoq'])
      expect(spellcheck.getCorrections('constructo')).toEqual(['constructor', 'constructoq'])
    })

    it('rank such words by frequency with a language model', function () {
      const spellcheck = new Spellcheck(['constructor', 'constructor', 'constructoq'], new LanguageModel())
      expect(spellcheck.getCorrections('constructo', 1, [])).toEqual(['constructor', 'constructoq'])
      const reversed = new Spellcheck(['constructoq', 'constructoq', 'constructor'], new LanguageModel())
      expect(reversed.getCorrections('constructo', 1, [])).toEqual(['constructoq', 'constructor'])
    })
  })

  describe('without a context', function () {
    it('gives the same corrections with and without a language model', function () {
      const plain = new Spellcheck(wordlist)
      const withModel = new Spellcheck(wordlist, trainedModel())
      expect(withModel.getCorrections('teh')).toEqual(plain.getCorrections('teh'))
      expect(withModel.getCorrections('teh', 2)).toEqual(plain.getCorrections('teh', 2))
      expect(withModel.getCorrections('teh', 1, null)).toEqual(plain.getCorrections('teh'))
      expect(plain.getCorrections('teh')[0]).toBe('the')
      expect(plain.getCorrections('teh').slice().sort()).toEqual(['tea', 'tee', 'ten', 'the'])
    })

    it('ignores the context when there is no language model', function () {
      const plain = new Spellcheck(wordlist)
      expect(plain.getCorrections('teh', 1, ['drink'])).toEqual(plain.getCorrections('teh'))
    })

    it('rejects a context that is not an array', function () {
      expect(() => new Spellcheck(wordlist).getCorrections('teh', 1, 'drink')).toThrowError(TypeError)
      expect(() => new Spellcheck(wordlist, trainedModel()).getCorrections('teh', 1, 'drink')).toThrowError(TypeError)
    })
  })

  describe('with a context', function () {
    let spellcheck
    beforeEach(function () {
      spellcheck = new Spellcheck(wordlist, trainedModel())
    })

    it('orders candidates by decreasing probability given the context', function () {
      expect(spellcheck.getCorrections('teh', 1, ['drink'])).toEqual(['tea', 'ten', 'the', 'tee'])
      expect(spellcheck.getCorrections('teh', 1, ['count'])).toEqual(['ten', 'tea', 'the', 'tee'])
    })

    it('breaks probability ties by edit distance, wordlist frequency and code unit order', function () {
      // Unseen context: 'tea' and 'ten' are equally probable (both were seen), 'the' and 'tee'
      // are equally probable (both unknown); 'the' is the more frequent of those two.
      expect(spellcheck.getCorrections('teh', 1, ['zzz'])).toEqual(['tea', 'ten', 'the', 'tee'])
      expect(spellcheck.getCorrections('teh', 1, [])).toEqual(['tea', 'ten', 'the', 'tee'])
    })

    it('only uses the last n - 1 context tokens', function () {
      expect(spellcheck.getCorrections('teh', 1, ['count', 'drink'])).toEqual(['tea', 'ten', 'the', 'tee'])
      expect(spellcheck.getCorrections('teh', 1, ['drink', 'count'])).toEqual(['ten', 'tea', 'the', 'tee'])
    })

    it('accepts start symbols in the context', function () {
      const corrections = spellcheck.getCorrections('teh', 1, [LanguageModel.START_SYMBOL])
      expect(corrections.slice().sort()).toEqual(['tea', 'tee', 'ten', 'the'])
      expect(corrections.slice(0, 2)).toEqual(['tea', 'ten'])
    })

    it('prefers a more probable candidate over a closer one', function () {
      const model = new LanguageModel({ n: 2 })
      model.addSentence(['a', 'cat'])
      model.addSentence(['a', 'cat'])
      model.addSentence(['a', 'cat'])
      model.train()
      const cats = new Spellcheck(['cat', 'coat', 'cot'], model)
      // 'cat' is two edits away from 'coot' but is far more probable after 'a'
      expect(cats.getCorrections('coot', 2, ['a'])).toEqual(['cat', 'coat', 'cot'])
      expect(cats.getCorrections('coot', 1, ['a'])).toEqual(['coat', 'cot'])
    })

    it('treats a correct input word as a candidate at distance 0', function () {
      const untrained = new Spellcheck(['tab', 'tabs', 'tabss', 'tabss', 'tabss'], new LanguageModel())
      expect(untrained.getCorrections('tab', 2, [])).toEqual(['tab', 'tabs', 'tabss'])
      expect(untrained.getCorrections('tabs', 1, [])).toEqual(['tabs', 'tabss', 'tab'])
      expect(untrained.getCorrections('tabss', 1, [])).toEqual(['tabss', 'tabs'])
    })

    it('uses a maximum distance of 1 by default', function () {
      const untrained = new Spellcheck(['tab', 'tabss'], new LanguageModel())
      expect(untrained.getCorrections('tabs', undefined, [])).toEqual(['tab', 'tabss'])
      expect(untrained.getCorrections('tabs', 0, [])).toEqual(['tab', 'tabss'])
      expect(untrained.getCorrections('tabsss', undefined, [])).toEqual(['tabss'])
    })

    it('falls back to distance, frequency and code unit order with an untrained model', function () {
      const untrained = new Spellcheck(['tac', 'tab', 'tad', 'tad'], new LanguageModel({ n: 3 }))
      expect(untrained.getCorrections('taz', 1, ['any', 'context'])).toEqual(['tad', 'tab', 'tac'])
    })

    it('returns every candidate once', function () {
      const duplicates = new Spellcheck(['tea', 'tea', 'ten'], trainedModel())
      expect(duplicates.getCorrections('teh', 2, ['drink'])).toEqual(['tea', 'ten'])
    })

    it('returns an empty list when no candidate is in the wordlist', function () {
      expect(spellcheck.getCorrections('xyzzy', 1, ['drink'])).toEqual([])
      expect(new Spellcheck([], trainedModel()).getCorrections('teh', 2, ['drink'])).toEqual([])
    })

    it('uses the scores of any object with a probability method', function () {
      const scores = { tee: 0.9, the: 0.1 }
      const contexts = []
      const scorer = {
        probability: function (word, context) {
          contexts.push(context)
          return scores[word] || 0
        }
      }
      const custom = new Spellcheck(wordlist, scorer)
      expect(custom.getCorrections('teh', 1, ['some', 'context'])).toEqual(['tee', 'the', 'tea', 'ten'])
      expect(contexts.length).toBe(4)
      expect(contexts[0]).toEqual(['some', 'context'])
    })
  })
})
