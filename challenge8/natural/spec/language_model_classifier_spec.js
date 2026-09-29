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

const fs = require('fs')
const os = require('os')
const path = require('path')

// The file storage backend reads FS_PATH when it is loaded, so provide a directory before the
// library is loaded when the environment does not define one.
if (!process.env.FS_PATH) {
  process.env.FS_PATH = fs.mkdtempSync(path.join(os.tmpdir(), 'natural-language-model-'))
}

const natural = require('../lib/natural')
// The storage module ({ StorageBackend, STORAGE_TYPES }) is loaded by path, as the io specs do,
// so this spec does not depend on how deeply the root export wraps it.
const storage = require('../lib/natural/util/storage/StorageBackend')
const Classifier = require('../lib/natural/classifiers/classifier')
const LanguageModelClassifier = natural.LanguageModelClassifier
const LanguageModel = natural.LanguageModel
const LancasterStemmer = natural.LancasterStemmer

const STORAGE_TIMEOUT = 10000
const DEFAULT_OPTIONS = { n: 3, discount: 0.75, unit: 'word', minCount: 1 }

function setupClassifier (options) {
  const classifier = new LanguageModelClassifier(options)
  classifier.addDocument('i fixed the box', 'computing')
  classifier.addDocument('i write code', 'computing')
  classifier.addDocument('nasty script code', 'computing')
  classifier.addDocument('write a book', 'literature')
  classifier.addDocument('read a book', 'literature')
  classifier.addDocument('study the books', 'literature')
  return classifier
}

describe('LanguageModelClassifier', function () {
  const filesToRemove = []
  let fileCounter = 0

  afterAll(function () {
    filesToRemove.forEach(function (file) {
      fs.rmSync(file, { force: true })
    })
  })

  // Ways of obtaining a copy of a classifier. Each returns a promise of the copy.
  function inMemoryCopy (classifier) {
    return Promise.resolve(classifier)
  }

  function restoredCopy (classifier) {
    return Promise.resolve(LanguageModelClassifier.restore(JSON.parse(JSON.stringify(classifier))))
  }

  function fileCopy (classifier, stemmer) {
    return new Promise(function (resolve, reject) {
      const filename = path.join(process.env.FS_PATH, 'language_model_classifier_' + process.pid + '_' + (fileCounter++) + '.json')
      filesToRemove.push(filename)
      classifier.save(filename, function (err) {
        if (err) {
          return reject(err)
        }
        LanguageModelClassifier.load(filename, stemmer, function (err, loaded) {
          if (err) {
            return reject(err)
          }
          resolve(loaded)
        })
      })
    })
  }

  async function storedCopy (classifier, stemmer) {
    const backend = new storage.StorageBackend()
    await backend.setStorageType(storage.STORAGE_TYPES.FILE)
    const key = await classifier.saveTo(backend)
    expect(typeof key).toBe('string')
    filesToRemove.push(path.join(process.env.FS_PATH, key + '.json'))
    return LanguageModelClassifier.loadFrom(key, stemmer, backend)
  }

  const copies = {
    'in memory': inMemoryCopy,
    'restored from JSON': restoredCopy,
    'saved to and loaded from a file': fileCopy,
    'stored in and loaded from a storage backend': storedCopy
  }

  it('is exported from the root of the library', function () {
    expect(typeof natural.LanguageModelClassifier).toBe('function')
    expect(new LanguageModelClassifier()).toEqual(jasmine.any(Classifier))
  })

  describe('options', function () {
    it('uses a trigram word model with discount 0.75 and minCount 1 by default', function () {
      const classifier = new LanguageModelClassifier()
      expect(classifier.options).toEqual(DEFAULT_OPTIONS)
      expect(classifier.models.size).toBe(0)
      expect(classifier.docs).toEqual([])
      expect(classifier.lastAdded).toBe(0)
    })

    it('accepts options in the constructor and through setOptions', function () {
      const classifier = new LanguageModelClassifier({ n: 2, unit: 'char' })
      expect(classifier.options).toEqual({ n: 2, discount: 0.75, unit: 'char', minCount: 1 })
      classifier.setOptions({ discount: 0.5, keepStops: true })
      expect(classifier.options).toEqual({ n: 2, discount: 0.5, unit: 'char', minCount: 1 })
      expect(classifier.keepStops).toBe(true)
      classifier.setOptions({ unit: 'word', n: 1, minCount: 3 })
      expect(classifier.options).toEqual({ n: 1, discount: 0.5, unit: 'word', minCount: 3 })
    })

    it('rejects invalid options', function () {
      expect(() => new LanguageModelClassifier({ n: 0 })).toThrowError(RangeError)
      expect(() => new LanguageModelClassifier({ discount: 2 })).toThrowError(RangeError)
      expect(() => new LanguageModelClassifier({ unit: 'syllable' })).toThrowError(RangeError)
      expect(() => new LanguageModelClassifier({ minCount: 0 })).toThrowError(RangeError)
      const classifier = new LanguageModelClassifier()
      expect(() => classifier.setOptions({ n: -1 })).toThrowError(RangeError)
      expect(() => classifier.setOptions(null)).toThrowError(TypeError)
      expect(classifier.options).toEqual(DEFAULT_OPTIONS)
    })

    it('does not allow the unit to change once documents have been added', function () {
      const classifier = new LanguageModelClassifier()
      classifier.addDocument('some text', 'label')
      expect(() => classifier.setOptions({ unit: 'char' })).toThrowError(Error)
      expect(classifier.options.unit).toBe('word')
      classifier.setOptions({ n: 2, minCount: 2 })
      expect(classifier.options.n).toBe(2)
      expect(classifier.options.minCount).toBe(2)
    })

    it('does not allow model options to change after training', function () {
      const classifier = setupClassifier()
      classifier.train()
      expect(() => classifier.setOptions({ n: 2 })).toThrowError(Error)
      expect(() => classifier.setOptions({ discount: 0.1 })).toThrowError(Error)
      expect(() => classifier.setOptions({ minCount: 2 })).toThrowError(Error)
      expect(classifier.options).toEqual(DEFAULT_OPTIONS)
      classifier.setOptions({ keepStops: true })
      expect(classifier.keepStops).toBe(true)
    })
  })

  describe('addDocument', function () {
    it('ignores documents without a classification or without tokens', function () {
      const classifier = new LanguageModelClassifier()
      classifier.addDocument('some text')
      classifier.addDocument('', 'label')
      classifier.addDocument([], 'label')
      expect(classifier.docs).toEqual([])
    })

    it('tokenises strings without stemming, stop word removal or case folding', function () {
      const classifier = new LanguageModelClassifier()
      classifier.addDocument('The Runners were running!', ' sports ')
      expect(classifier.docs).toEqual([{ label: 'sports', text: ['The', 'Runners', 'were', 'running'] }])
    })

    it('splits strings into characters for the char unit', function () {
      const classifier = new LanguageModelClassifier({ unit: 'char' })
      classifier.addDocument('ab c', 'label')
      expect(classifier.docs).toEqual([{ label: 'label', text: ['a', 'b', ' ', 'c'] }])
    })

    it('keeps its own copy of token arrays', function () {
      const classifier = new LanguageModelClassifier()
      const tokens = ['write', 'code']
      classifier.addDocument(tokens, 'computing')
      tokens.push('changed')
      expect(classifier.docs).toEqual([{ label: 'computing', text: ['write', 'code'] }])
    })

    it('counts features, including tokens named like Object.prototype members', function () {
      const classifier = new LanguageModelClassifier()
      classifier.addDocument(['constructor', 'code'], 'computing')
      classifier.addDocument(['constructor', 'toString'], 'computing')
      expect(classifier.features.constructor).toBe(2)
      expect(classifier.features.code).toBe(1)
      expect(classifier.features.toString).toBe(1)
      expect(classifier.features.valueOf).toBeUndefined()
    })

    it('rejects documents that are neither strings nor arrays of strings', function () {
      const classifier = new LanguageModelClassifier()
      expect(() => classifier.addDocument(42, 'label')).toThrowError(TypeError)
      expect(() => classifier.addDocument(['a', 1], 'label')).toThrowError(TypeError)
    })
  })

  describe('ignores the stemmer', function () {
    it('uses the tokens as they are', function () {
      const classifier = new LanguageModelClassifier()
      classifier.addDocument('Running quickly', 'motion')
      classifier.addDocument('Sleeping soundly', 'rest')
      expect(classifier.docs).toEqual([
        { label: 'motion', text: ['Running', 'quickly'] },
        { label: 'rest', text: ['Sleeping', 'soundly'] }
      ])
      classifier.train()
      expect(classifier.classify('Running')).toBe('motion')
      expect(classifier.classify(['Sleeping'])).toBe('rest')
      expect(classifier.getClassifications('running')).toEqual(classifier.getClassifications('Walking'))
    })

    Object.keys(copies).forEach(function (name) {
      it('when a stemmer is passed while being copied ' + name, async function () {
        const classifier = new LanguageModelClassifier()
        classifier.addDocument('Running quickly', 'motion')
        classifier.addDocument('Sleeping soundly', 'rest')
        classifier.train()
        const copy = await copies[name](classifier, LancasterStemmer)
        expect(copy.docs).toEqual(classifier.docs)
        expect(copy.classify('Running')).toBe('motion')
        copy.addDocument('Walking slowly', 'motion')
        expect(copy.docs[2]).toEqual({ label: 'motion', text: ['Walking', 'slowly'] })
      }, STORAGE_TIMEOUT)
    })
  })

  describe('train', function () {
    it('emits events for every document and when training is done', function () {
      const classifier = setupClassifier()
      const events = []
      let done = null
      classifier.on('trainedWithDocument', function (event) { events.push(event) })
      classifier.on('doneTraining', function (value) { done = value })
      classifier.train()
      expect(events.length).toBe(6)
      expect(events[0]).toEqual({ index: 0, total: 6, doc: { label: 'computing', text: ['i', 'fixed', 'the', 'box'] } })
      expect(events[5]).toEqual({ index: 5, total: 6, doc: { label: 'literature', text: ['study', 'the', 'books'] } })
      expect(done).toBe(true)
      expect(classifier.lastAdded).toBe(6)
    })

    it('builds one trained language model per label with the classifier options', function () {
      const classifier = setupClassifier({ n: 2, discount: 0.5, minCount: 2 })
      classifier.train()
      expect(Array.from(classifier.models.keys())).toEqual(['computing', 'literature'])
      classifier.models.forEach(function (model) {
        expect(model).toEqual(jasmine.any(LanguageModel))
        expect(model.trained).toBe(true)
        expect(model.n).toBe(2)
        expect(model.discount).toBe(0.5)
        expect(model.unit).toBe('word')
        expect(model.minCount).toBe(2)
        expect(model.sentences.length).toBe(3)
      })
      expect(classifier.models.get('computing').vocabulary()).toEqual(['</s>', 'code', 'i'])
      expect(classifier.models.get('computing').sentences[1]).toEqual(['i', 'write', 'code'])
    })

    it('only trains with the documents added since the last call', function () {
      const classifier = setupClassifier()
      classifier.train()
      const events = []
      classifier.on('trainedWithDocument', function (event) { events.push(event) })
      classifier.train()
      expect(events).toEqual([])
      classifier.addDocument('debug the code', 'computing')
      classifier.train()
      expect(events.length).toBe(1)
      expect(events[0].index).toBe(6)
      expect(events[0].total).toBe(7)
      expect(classifier.models.get('computing').sentences.length).toBe(4)
      expect(classifier.models.get('literature').sentences.length).toBe(3)
    })

    it('classifies text and token arrays', function () {
      const classifier = setupClassifier()
      classifier.train()
      expect(classifier.classify('i write some code')).toBe('computing')
      expect(classifier.classify(['read', 'the', 'book'])).toBe('literature')
      expect(classifier.classify('study the books')).toBe('literature')
      expect(classifier.classify('fix the box')).toBe('computing')
    })

    it('returns normalised classifications sorted by decreasing value', function () {
      const classifier = setupClassifier()
      classifier.train()
      const classifications = classifier.getClassifications('write a book')
      expect(classifications.map(function (c) { return c.label })).toEqual(['literature', 'computing'])
      expect(classifications[0].value).toBeGreaterThan(classifications[1].value)
      expect(classifications[0].value + classifications[1].value).toBeCloseTo(1, 12)
      expect(classifications[1].value).toBeGreaterThan(0)
    })

    it('computes the posterior from the label prior and the language model', function () {
      // Character unigram models with discount 0.5: for label A ('aa') P(a) = 11/18 and for
      // label B ('bb') P(a) = 2/18; the end symbol probabilities are equal, so P(A | 'a') = 11/13.
      const classifier = new LanguageModelClassifier({ unit: 'char', n: 1, discount: 0.5 })
      classifier.addDocument('aa', 'A')
      classifier.addDocument('bb', 'B')
      classifier.train()
      const classifications = classifier.getClassifications('a')
      expect(classifications.map(function (c) { return c.label })).toEqual(['A', 'B'])
      expect(classifications[0].value).toBeCloseTo(11 / 13, 12)
      expect(classifications[1].value).toBeCloseTo(2 / 13, 12)
      expect(classifier.models.get('A').probability('a')).toBeCloseTo(11 / 18, 12)
      expect(classifier.models.get('B').probability('a')).toBeCloseTo(2 / 18, 12)
    })

    it('takes the label prior into account', function () {
      // With discount 0 both unigram models assign the same probabilities, so only the
      // number of documents per label differs
      const classifier = new LanguageModelClassifier({ n: 1, discount: 0 })
      classifier.addDocument(['x'], 'frequent')
      classifier.addDocument(['x'], 'frequent')
      classifier.addDocument(['x'], 'frequent')
      classifier.addDocument(['x'], 'rare')
      classifier.train()
      const classifications = classifier.getClassifications(['x'])
      expect(classifications[0].label).toBe('frequent')
      expect(classifications[0].value).toBeCloseTo(0.75, 12)
      expect(classifications[1].value).toBeCloseTo(0.25, 12)
    })

    it('breaks ties between equally probable labels by label', function () {
      const classifier = new LanguageModelClassifier()
      classifier.addDocument('same text', 'zulu')
      classifier.addDocument('same text', 'alpha')
      classifier.train()
      const classifications = classifier.getClassifications('same text')
      expect(classifications.map(function (c) { return c.label })).toEqual(['alpha', 'zulu'])
      expect(classifications[0].value).toBeCloseTo(0.5, 12)
      expect(classifier.classify('same text')).toBe('alpha')
    })

    it('classifies nothing before training', function () {
      const classifier = setupClassifier()
      expect(classifier.getClassifications('write code')).toEqual([])
      expect(classifier.classify('write code')).toBeUndefined()
    })

    it('rejects observations that are neither strings nor arrays of strings', function () {
      const classifier = setupClassifier()
      classifier.train()
      expect(() => classifier.classify(42)).toThrowError(TypeError)
      expect(() => classifier.getClassifications([1])).toThrowError(TypeError)
    })
  })

  describe('removeDocument', function () {
    it('removes the last matching document and its features', function () {
      const classifier = new LanguageModelClassifier()
      classifier.addDocument('write code', 'computing')
      classifier.addDocument('write a book', 'literature')
      classifier.addDocument('write code', 'computing')
      classifier.removeDocument('write code', ' computing ')
      expect(classifier.docs).toEqual([
        { label: 'computing', text: ['write', 'code'] },
        { label: 'literature', text: ['write', 'a', 'book'] }
      ])
      expect(classifier.features.write).toBe(2)
      expect(classifier.features.code).toBe(1)
      classifier.removeDocument(['write', 'code'], 'computing')
      expect(classifier.docs).toEqual([{ label: 'literature', text: ['write', 'a', 'book'] }])
      expect(classifier.features.write).toBe(1)
      expect(classifier.features.code).toBeUndefined()
    })

    it('does nothing when text or label do not match', function () {
      const classifier = setupClassifier()
      classifier.removeDocument('write a book', 'computing')
      classifier.removeDocument('something else', 'literature')
      classifier.removeDocument(['write a book'], 'literature')
      expect(classifier.docs.length).toBe(6)
    })

    it('keeps incremental training aligned when a trained document is removed', function () {
      const classifier = setupClassifier()
      classifier.train()
      classifier.removeDocument('i write code', 'computing')
      expect(classifier.docs.length).toBe(5)
      expect(classifier.lastAdded).toBe(5)
      classifier.addDocument('new document', 'computing')
      const events = []
      classifier.on('trainedWithDocument', function (event) { events.push(event) })
      classifier.train()
      expect(events.length).toBe(1)
      expect(events[0].doc).toEqual({ label: 'computing', text: ['new', 'document'] })
    })

    it('does not change the models until the classifier is retrained', function () {
      const classifier = setupClassifier()
      classifier.train()
      const before = classifier.getClassifications('i write code')
      classifier.removeDocument('i write code', 'computing')
      classifier.removeDocument('nasty script code', 'computing')
      expect(classifier.getClassifications('i write code')).toEqual(before)
      classifier.retrain()
      expect(classifier.models.get('computing').sentences.length).toBe(1)
      expect(classifier.getClassifications('i write code')).not.toEqual(before)
      expect(classifier.classify('write a book')).toBe('literature')
    })
  })

  describe('retrain', function () {
    it('rebuilds the models from all documents and emits all events again', function () {
      const classifier = setupClassifier()
      classifier.train()
      const events = []
      classifier.on('trainedWithDocument', function (event) { events.push(event) })
      classifier.retrain()
      expect(events.length).toBe(6)
      expect(classifier.lastAdded).toBe(6)
      expect(classifier.models.size).toBe(2)
      expect(classifier.classify('i write code')).toBe('computing')
    })
  })

  describe('toJSON and restore', function () {
    it('serialises options, documents, features and models', function () {
      const classifier = setupClassifier({ n: 2, minCount: 2 })
      classifier.setOptions({ keepStops: true })
      classifier.train()
      const json = classifier.toJSON()
      expect(json.options).toEqual({ n: 2, discount: 0.75, unit: 'word', minCount: 2 })
      expect(json.keepStops).toBe(true)
      expect(json.docs).toEqual(classifier.docs)
      expect(json.features).toEqual(Object.assign({}, classifier.features))
      expect(json.lastAdded).toBe(6)
      expect(json.models.length).toBe(2)
      expect(json.models[0][0]).toBe('computing')
      expect(json.models[0][1]).toEqual(classifier.models.get('computing').toJSON())
      expect(json.models[1][0]).toBe('literature')
      expect(JSON.parse(JSON.stringify(classifier))).toEqual(JSON.parse(JSON.stringify(json)))
    })

    it('does not share state with the JSON it produces or restores from', function () {
      const classifier = setupClassifier()
      classifier.train()
      const json = classifier.toJSON()
      json.docs[0].text[0] = 'changed'
      json.models[0][1].sentences[0][0] = 'changed'
      json.features.i = 99
      expect(classifier.docs[0].text[0]).toBe('i')
      expect(classifier.models.get('computing').sentences[0][0]).toBe('i')
      expect(classifier.features.i).toBe(2)

      const state = classifier.toJSON()
      const copy = LanguageModelClassifier.restore(state)
      state.docs[0].text[0] = 'changed'
      state.models[0][1].sentences[0][0] = 'changed'
      state.docs.push({ label: 'x', text: ['y'] })
      expect(copy.docs).toEqual(classifier.docs)
      expect(copy.models.get('computing').sentences).toEqual(classifier.models.get('computing').sentences)
      copy.addDocument('only in the copy', 'computing')
      expect(classifier.docs.length).toBe(6)
    })

    it('rejects malformed input', function () {
      expect(() => LanguageModelClassifier.restore(null)).toThrowError(TypeError)
      expect(() => LanguageModelClassifier.restore({})).toThrowError(TypeError)
      expect(() => LanguageModelClassifier.restore({ docs: [], models: [], options: { n: 0 } })).toThrowError(RangeError)
    })
  })

  describe('copies keep models and documents', function () {
    async function assertKeepsModelsAndDocuments (copy) {
      const classifier = setupClassifier()

      // An untrained copy keeps the documents and is still untrained
      const untrained = await copy(classifier)
      expect(untrained.docs).toEqual(classifier.docs)
      expect(untrained.models.size).toBe(0)
      expect(untrained.lastAdded).toBe(0)
      expect(untrained.getClassifications('write code')).toEqual([])

      // Training the copy behaves like training the original, including its events
      const events = []
      let done = null
      untrained.on('trainedWithDocument', function (event) { events.push(event) })
      untrained.on('doneTraining', function (value) { done = value })
      untrained.train()
      expect(events.length).toBe(6)
      expect(events[0].index).toBe(0)
      expect(events[0].total).toBe(6)
      expect(events[5].doc).toEqual({ label: 'literature', text: ['study', 'the', 'books'] })
      expect(done).toBe(true)
      classifier.train()
      expect(untrained.classify('write some code')).toBe('computing')
      expect(untrained.getClassifications('read a book')).toEqual(classifier.getClassifications('read a book'))

      // A trained copy keeps the models and the training state
      const trained = await copy(classifier)
      expect(trained.docs).toEqual(classifier.docs)
      expect(trained.lastAdded).toBe(6)
      expect(Array.from(trained.models.keys())).toEqual(['computing', 'literature'])
      expect(trained.models.get('literature').trained).toBe(true)
      expect(trained.getClassifications('i write code')).toEqual(classifier.getClassifications('i write code'))
      expect(trained.classify('study the book')).toBe('literature')

      // Removing a document and retraining works the same on the copy
      trained.removeDocument('i write code', 'computing')
      classifier.removeDocument('i write code', 'computing')
      expect(trained.docs.length).toBe(5)
      expect(trained.lastAdded).toBe(5)
      expect(trained.classify('i write code')).toBe('computing')
      trained.retrain()
      classifier.retrain()
      expect(trained.models.get('computing').sentences.length).toBe(2)
      expect(trained.getClassifications('i write code')).toEqual(classifier.getClassifications('i write code'))

      // A real copy and the original are independent afterwards
      if (trained !== classifier) {
        trained.addDocument('brand new document', 'computing')
        trained.train()
        expect(classifier.docs.length).toBe(5)
        expect(classifier.models.get('computing').sentences.length).toBe(2)
        expect(trained.docs.length).toBe(6)
        expect(trained.models.get('computing').sentences.length).toBe(3)
      }
    }

    Object.keys(copies).forEach(function (name) {
      it('when ' + name, async function () {
        await assertKeepsModelsAndDocuments(copies[name])
      }, STORAGE_TIMEOUT)
    })
  })

  describe('labels named like Object.prototype members', function () {
    const labels = ['constructor', '__proto__', 'hasOwnProperty', 'toString', 'valueOf', '__defineGetter__']

    function setupPropertyNameClassifier () {
      const classifier = new LanguageModelClassifier()
      labels.forEach(function (label) {
        classifier.addDocument([label, 'first'], label)
        classifier.addDocument([label, 'second'], label)
      })
      return classifier
    }

    it('are kept apart in memory', function () {
      const classifier = setupPropertyNameClassifier()
      expect(classifier.docs.length).toBe(12)
      expect(classifier.features.constructor).toBe(2)
      expect(classifier.features.first).toBe(6)
      expect(Object.entries(classifier.features)).toContain(['__proto__', 2])
      expect(JSON.parse(JSON.stringify(classifier)).features).toEqual({ ...classifier.features })
      classifier.train()
      expect(Array.from(classifier.models.keys())).toEqual(labels)
      labels.forEach(function (label) {
        expect(classifier.classify([label, 'first'])).toBe(label)
        expect(classifier.models.get(label).sentences.length).toBe(2)
      })
      expect(classifier.getClassifications(['toString', 'second']).length).toBe(labels.length)
    })

    Object.keys(copies).forEach(function (name) {
      it('survive being copied ' + name, async function () {
        const classifier = setupPropertyNameClassifier()
        classifier.train()
        const copy = await copies[name](classifier)
        expect(copy.docs).toEqual(classifier.docs)
        expect(Array.from(copy.models.keys())).toEqual(labels)
        expect(copy.features.constructor).toBe(2)
        expect(Object.entries(copy.features)).toContain(['__proto__', 2])
        expect(copy.features.first).toBe(6)
        labels.forEach(function (label) {
          expect(copy.classify([label, 'second'])).toBe(label)
        })
        expect(copy.getClassifications(['valueOf', 'first']).map(function (c) { return c.label }).sort())
          .toEqual(labels.slice().sort())
        copy.removeDocument(['constructor', 'first'], 'constructor')
        copy.retrain()
        expect(copy.models.get('constructor').sentences).toEqual([['constructor', 'second']])
        if (copy !== classifier) {
          expect(classifier.models.get('constructor').sentences.length).toBe(2)
        }
      }, STORAGE_TIMEOUT)
    })
  })

  describe('options survive copies', function () {
    Object.keys(copies).forEach(function (name) {
      it('when ' + name, async function () {
        const characters = new LanguageModelClassifier({ unit: 'char', n: 1, discount: 0.5 })
        characters.setOptions({ keepStops: true })
        const charactersCopy = await copies[name](characters)
        expect(charactersCopy.options).toEqual({ n: 1, discount: 0.5, unit: 'char', minCount: 1 })
        expect(charactersCopy.keepStops).toBe(true)
        expect(charactersCopy.models.size).toBe(0)
        charactersCopy.addDocument('aa', 'A')
        charactersCopy.addDocument('bb', 'B')
        charactersCopy.train()
        const classifications = charactersCopy.getClassifications('a')
        expect(classifications.map(function (c) { return c.label })).toEqual(['A', 'B'])
        expect(classifications[0].value).toBeCloseTo(11 / 13, 12)
        expect(classifications[1].value).toBeCloseTo(2 / 13, 12)
        expect(charactersCopy.models.get('A').unit).toBe('char')

        const words = new LanguageModelClassifier({ n: 2, minCount: 2 })
        const wordsCopy = await copies[name](words)
        expect(wordsCopy.options).toEqual({ n: 2, discount: 0.75, unit: 'word', minCount: 2 })
        wordsCopy.addDocument('read the book the book', 'literature')
        wordsCopy.addDocument('write the code the code', 'computing')
        wordsCopy.train()
        expect(wordsCopy.classify('the book')).toBe('literature')
        expect(wordsCopy.models.get('literature').n).toBe(2)
        expect(wordsCopy.models.get('literature').minCount).toBe(2)
        expect(wordsCopy.models.get('literature').vocabulary()).toEqual(['</s>', 'book', 'the'])
        expect(wordsCopy.models.get('literature').sentences).toEqual([['read', 'the', 'book', 'the', 'book']])
      }, STORAGE_TIMEOUT)
    })
  })

  describe('file and storage backend persistence', function () {
    it('reports file errors through the callback', function (done) {
      LanguageModelClassifier.load(path.join(process.env.FS_PATH, 'does-not-exist', 'classifier.json'), null, function (err, classifier) {
        expect(err).not.toBeNull()
        expect(classifier).toBeUndefined()
        done()
      })
    })

    it('saves to and loads from a storage backend', async function () {
      const classifier = setupClassifier()
      classifier.train()
      const backend = new storage.StorageBackend()
      await backend.setStorageType(storage.STORAGE_TYPES.FILE)
      const key = await classifier.saveTo(backend)
      expect(typeof key).toBe('string')
      expect(key.length).toBeGreaterThan(0)
      filesToRemove.push(path.join(process.env.FS_PATH, key + '.json'))
      const loaded = await LanguageModelClassifier.loadFrom(key, LancasterStemmer, backend)
      expect(loaded).toEqual(jasmine.any(LanguageModelClassifier))
      expect(loaded).not.toBe(classifier)
      expect(loaded.options).toEqual(classifier.options)
      expect(loaded.docs).toEqual(classifier.docs)
      expect(loaded.lastAdded).toBe(6)
      expect(loaded.getClassifications('write a book')).toEqual(classifier.getClassifications('write a book'))
    }, STORAGE_TIMEOUT)

    it('rejects when the key is unknown to the storage backend', async function () {
      const backend = new storage.StorageBackend()
      await backend.setStorageType(storage.STORAGE_TYPES.FILE)
      await expectAsync(LanguageModelClassifier.loadFrom('missing-key', null, backend)).toBeRejected()
    }, STORAGE_TIMEOUT)
  })
})
