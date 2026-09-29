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

// Exercises the declared LanguageModel, LanguageModelClassifier and Spellcheck API under
// strict TypeScript, so the declarations keep matching the runtime signatures.

import { LanguageModel, LanguageModelClassifier, LancasterStemmer, Spellcheck } from 'lib/natural'
import type { LanguageModelJSON, StorageBackend } from 'lib/natural'

// A minimal in-memory object with the store/retrieve shape of a storage backend
function memoryBackend (): StorageBackend {
  const objects = new Map<string, string>()
  const backend = {
    store: async (object: unknown): Promise<string> => {
      const key = 'key-' + String(objects.size)
      objects.set(key, JSON.stringify(object))
      return key
    },
    retrieve: async (key: string): Promise<unknown> => {
      const data = objects.get(key)
      if (data === undefined) {
        throw new Error('unknown key ' + key)
      }
      return JSON.parse(data)
    }
  }
  return backend as unknown as StorageBackend
}

describe('LanguageModel declarations', function () {
  it('declare the language model API', function () {
    const model: LanguageModel = new LanguageModel({ n: 2, discount: 0.5, mode: 'word' })
    const added: boolean = model.addSentence('drink tea')
    model.addSentence(['count', 'ten'])
    model.train()
    const probability: number = model.probability('tea', ['drink'])
    const unigram: number = model.probability('tea')
    const logProbability: number = model.logProbability('drink tea')
    const json: LanguageModelJSON = model.toJSON()
    const copy: LanguageModel = LanguageModel.restore(json)
    const vocabulary: string[] = copy.vocabulary()
    const tokens: string[] = LanguageModel.tokenize('drink tea', 'character')
    const start: string = LanguageModel.START_SYMBOL

    expect(added).toBe(true)
    expect(probability).toBeGreaterThan(unigram)
    expect(logProbability).toBeLessThan(0)
    expect(vocabulary).toContain('tea')
    expect(tokens.length).toBe(9)
    expect(start).toBe('<s>')
    expect(copy.n).toBe(2)
    expect(copy.trained).toBe(true)
  })

  it('declare the classifier API including its storage methods', async function () {
    const classifier = new LanguageModelClassifier(LancasterStemmer, { n: 2, mode: 'word' })
    classifier.setOptions({ discount: 0.5, keepStops: true })
    classifier.addDocument('write the code', 'computing')
    classifier.addDocument(['read', 'the', 'book'], 'literature')
    classifier.train()
    const label: string = classifier.classify('the code')
    const classifications = classifier.getClassifications('the book')
    const models: Map<string, LanguageModel> = classifier.models
    const backend = memoryBackend()
    const key: string = await classifier.saveTo(backend)
    const loaded: LanguageModelClassifier = await LanguageModelClassifier.loadFrom(key, LancasterStemmer, backend)
    const restored: LanguageModelClassifier = LanguageModelClassifier.restore(classifier.toJSON(), null)

    expect(label).toBe('computing')
    expect(classifications[0].label).toBe('literature')
    expect(models.size).toBe(2)
    expect(loaded.classify('the book')).toBe('literature')
    expect(loaded.options.discount).toBe(0.5)
    expect(restored.options.n).toBe(2)
  })

  it('declare the contextual spellcheck API', function () {
    const model = new LanguageModel({ n: 2 })
    model.addSentence(['drink', 'tea'])
    model.train()
    const spellcheck = new Spellcheck(['tea', 'ten'], model)
    const corrections: string[] = spellcheck.getCorrections('teh', 1, ['drink'])
    expect(corrections).toEqual(['tea', 'ten'])
    expect(new Spellcheck(['tea']).getCorrections('teh')).toEqual(['tea'])
    expect(new Spellcheck(['tea'], null).isCorrect('tea')).toBe(true)
  })
})
