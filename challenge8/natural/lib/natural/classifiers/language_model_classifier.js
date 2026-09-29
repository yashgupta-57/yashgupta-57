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

const Classifier = require('./classifier')
const LanguageModel = require('../language_model/language_model')

const MODEL_OPTION_KEYS = ['n', 'discount', 'unit', 'minCount']

// Validates language model options and returns the normalised values
function readModelOptions (options) {
  const model = new LanguageModel(options)
  return { n: model.n, discount: model.discount, unit: model.unit, minCount: model.minCount }
}

function sameTokens (a, b) {
  if (a.length !== b.length) {
    return false
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false
    }
  }
  return true
}

function compareLabels (a, b) {
  const left = String(a)
  const right = String(b)
  if (left < right) {
    return -1
  }
  return left > right ? 1 : 0
}

// Classifier that trains one Kneser-Ney language model per label and classifies text by the
// posterior probability of each label (the label's prior times the probability of the text
// under the label's model). Unlike the other classifiers, text is never stemmed: tokens are used
// as they are, and the stemmer parameter of load and loadFrom is accepted for compatibility only.
class LanguageModelClassifier extends Classifier {
  // options: { n, discount, unit, minCount } (see LanguageModel) plus the base keepStops option
  constructor (options) {
    super(null)
    this.features = Object.create(null)
    this.options = readModelOptions(options)
    this.models = new Map()
  }

  // n, discount, unit and minCount can be set until the classifier is trained; unit can only be
  // set while no documents have been added, because documents are tokenised when they are added.
  setOptions (options) {
    if (options === null || typeof options !== 'object') {
      throw new TypeError('options must be an object')
    }
    super.setOptions(options)
    const requested = {}
    let changed = false
    for (const key of MODEL_OPTION_KEYS) {
      if (options[key] !== undefined) {
        requested[key] = options[key]
        changed = true
      }
    }
    if (!changed) {
      return
    }
    if (this.models.size > 0) {
      throw new Error('n, discount, unit and minCount cannot be changed after training; create a new classifier instead')
    }
    const next = readModelOptions(Object.assign({}, this.options, requested))
    if (next.unit !== this.options.unit && this.docs.length > 0) {
      throw new Error('unit cannot be changed once documents have been added')
    }
    this.options = next
  }

  tokenize (text) {
    return LanguageModel.tokenize(text, this.options.unit)
  }

  toTokens (text) {
    if (typeof text === 'string') {
      return this.tokenize(text)
    }
    if (Array.isArray(text)) {
      for (const token of text) {
        if (typeof token !== 'string') {
          throw new TypeError('text must be a string or an array of strings')
        }
      }
      return text.slice()
    }
    throw new TypeError('text must be a string or an array of strings')
  }

  addDocument (text, classification) {
    // Ignore further processing if classification is undefined
    if (typeof classification === 'undefined') {
      return
    }
    if (typeof classification === 'string') {
      classification = classification.trim()
    }
    const tokens = this.toTokens(text)
    if (tokens.length === 0) {
      // ignore empty documents
      return
    }
    this.docs.push({ label: classification, text: tokens })
    for (const token of tokens) {
      this.features[token] = (this.features[token] || 0) + 1
    }
  }

  // Removes the last document with the given text and label. Trained models are not changed
  // until retrain() is called.
  removeDocument (text, classification) {
    if (typeof classification === 'string') {
      classification = classification.trim()
    }
    const tokens = this.toTokens(text)
    let position = -1
    for (let i = 0; i < this.docs.length; i++) {
      const doc = this.docs[i]
      if (doc.label === classification && sameTokens(doc.text, tokens)) {
        position = i
      }
    }
    if (position === -1) {
      return
    }
    this.docs.splice(position, 1)
    for (const token of tokens) {
      if (this.features[token] > 1) {
        this.features[token]--
      } else {
        delete this.features[token]
      }
    }
    if (position < this.lastAdded) {
      this.lastAdded--
    }
  }

  // Trains the models with the documents added since the last call
  train () {
    const totalDocs = this.docs.length
    const updated = new Set()
    for (let i = this.lastAdded; i < totalDocs; i++) {
      const doc = this.docs[i]
      let model = this.models.get(doc.label)
      if (model === undefined) {
        model = new LanguageModel(this.options)
        this.models.set(doc.label, model)
      }
      model.addSentence(doc.text)
      updated.add(model)
      this.emit('trainedWithDocument', { index: i, total: totalDocs, doc })
      this.lastAdded++
    }
    for (const model of updated) {
      model.train()
    }
    this.emit('doneTraining', true)
  }

  retrain () {
    this.models = new Map()
    this.lastAdded = 0
    this.train()
  }

  // Returns [{ label, value }] sorted by decreasing value, where value is the posterior
  // probability of the label. Ties are broken by label.
  getClassifications (observation) {
    const tokens = this.toTokens(observation)
    if (this.models.size === 0) {
      return []
    }
    let totalSentences = 0
    for (const model of this.models.values()) {
      totalSentences += model.sentences.length
    }
    const scores = []
    let max = -Infinity
    for (const [label, model] of this.models) {
      const score = Math.log(model.sentences.length / totalSentences) + model.logProbability(tokens)
      scores.push({ label, score })
      max = Math.max(max, score)
    }
    const weights = scores.map(entry => (max === -Infinity ? 1 : Math.exp(entry.score - max)))
    const sum = weights.reduce((total, weight) => total + weight, 0)
    return scores
      .map((entry, i) => ({ label: entry.label, value: weights[i] / sum }))
      .sort((a, b) => (b.value - a.value) || compareLabels(a.label, b.label))
  }

  classify (observation) {
    const classifications = this.getClassifications(observation)
    return classifications.length > 0 ? classifications[0].label : undefined
  }

  toJSON () {
    return {
      options: Object.assign({}, this.options),
      keepStops: !!this.keepStops,
      docs: this.docs.map(doc => ({ label: doc.label, text: doc.text.slice() })),
      // Spread defines own properties, so a feature called '__proto__' survives the copy
      features: { ...this.features },
      lastAdded: this.lastAdded,
      models: Array.from(this.models, ([label, model]) => [label, model.toJSON()])
    }
  }

  // Builds an independent classifier from the output of toJSON()
  static restore (classifier) {
    if (classifier === null || typeof classifier !== 'object' ||
        !Array.isArray(classifier.docs) || !Array.isArray(classifier.models)) {
      throw new TypeError('classifier must be an object created by LanguageModelClassifier#toJSON')
    }
    const restored = new LanguageModelClassifier(classifier.options)
    restored.keepStops = !!classifier.keepStops
    restored.docs = classifier.docs.map(doc => ({ label: doc.label, text: restored.toTokens(doc.text) }))
    restored.features = Object.assign(Object.create(null), classifier.features)
    restored.lastAdded = Number.isInteger(classifier.lastAdded) ? classifier.lastAdded : 0
    restored.models = new Map(classifier.models.map(([label, json]) => [label, LanguageModel.restore(json)]))
    return restored
  }

  // The stemmer has the same position as in the other classifiers and is ignored
  static load (filename, stemmer, callback) {
    Classifier.load(filename, stemmer, function (err, classifier) {
      if (err) {
        return callback(err)
      }
      callback(err, LanguageModelClassifier.restore(classifier))
    })
  }

  // The stemmer has the same position as in the other classifiers and is ignored
  static async loadFrom (key, stemmer, storageBackend) {
    const classifier = await Classifier.loadFrom(key, storageBackend)
    return LanguageModelClassifier.restore(classifier)
  }
}

module.exports = LanguageModelClassifier
