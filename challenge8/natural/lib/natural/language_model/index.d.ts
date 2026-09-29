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

export type LanguageModelMode = 'word' | 'character'

export interface LanguageModelOptions {
  /** Order of the model (default 3) */
  n?: number
  /** Kneser-Ney discount between 0 and 1 (default 0.75) */
  discount?: number
  /** How strings are split into tokens (default 'word') */
  mode?: LanguageModelMode
}

export interface LanguageModelJSON {
  n: number
  discount: number
  mode: LanguageModelMode
  trained: boolean
  sentences: string[][]
}

export declare class LanguageModel {
  static readonly START_SYMBOL: string
  static readonly END_SYMBOL: string

  readonly n: number
  readonly discount: number
  readonly mode: LanguageModelMode
  readonly sentences: ReadonlyArray<readonly string[]>
  readonly trained: boolean

  constructor (options?: LanguageModelOptions)
  static tokenize (text: string, mode?: LanguageModelMode): string[]
  tokenize (text: string): string[]
  addSentence (sentence: string | string[]): boolean
  train (): void
  probability (word: string, context?: string[]): number
  logProbability (sentence: string | string[]): number
  vocabulary (): string[]
  toJSON (): LanguageModelJSON
  static restore (json: LanguageModelJSON): LanguageModel
}
