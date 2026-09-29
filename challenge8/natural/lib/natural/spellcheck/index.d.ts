/*
Copyright (c) 2022,
Dylan R. E. Moonfire <https://github.com/dmoonfire>,
Emily Marigold Klassen <https://github.com/forivall>,
Hugo W.L. ter Doest <https://github.com/Hugo-ter-Doest>

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

/** Anything that can score a word given the tokens preceding it, such as LanguageModel */
export interface SpellcheckLanguageModel {
  probability: (word: string, context: string[]) => number
}

export declare class Spellcheck {
  constructor (wordlist: string[], languageModel?: SpellcheckLanguageModel | null)
  isCorrect (word: string): boolean
  /**
   * Suggested corrections, best first. With a language model and a context (the tokens
   * preceding the word) candidates are ordered by decreasing probability under the model,
   * then by increasing edit distance, decreasing wordlist frequency and code unit order.
   */
  getCorrections (word: string, maxDistance?: number, context?: string[] | null): string[]
  edits (word: string): string[]
  editsWithMaxDistance (word: string, maxDistance: number): string[]
}
