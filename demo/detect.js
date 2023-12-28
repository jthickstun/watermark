import { MersenneTwister } from "./mersenne.js";

function permutation_test(tokens, seed, N, k, n_runs = 99) {
  const vocab_size = 32000

  var rng = new MersenneTwister(seed);
  var u = Array.from({length: N * vocab_size}, () => rng.random())

  var test_result = test_stat(tokens, u, N, k);

  const uniqueTokens = [...new Set(tokens)];
  tokens = tokens.map((token) => uniqueTokens.indexOf(token));

  var p_val = 0
  var u_alternative = new Array(N*uniqueTokens.length);
  var null_result;
  for (var run = 0; run < n_runs; run++) {
    for (let p = 0; p < N * uniqueTokens.length; p++) {
      u_alternative[p] = Math.random()
    }

    null_result = test_stat(tokens, u_alternative, N, k);
    p_val += (null_result <= test_result) ? 1.0 : 0;
  }

  return (p_val+1)/(n_runs+1);
};

window.permutation_test = permutation_test

function test_stat(tokens, u, N, k) {
  const vocab = u.length / N;
  let A = new Array(tokens.length-(k-1));
  let sub = new Array(vocab*k);
  for (let i = 0; i < A.length; i++) {
    A[i] = new Array(N)
    for (let j = 0; j < N; j++) {
      for (let p = 0; p < vocab*k; p++) {
        sub[p] = u[(vocab*j + p) % (vocab*N)];
      }
      A[i][j] = levenshtein(tokens.slice(i, i+k), sub, vocab)
    }
  }

  // min-cost for each alignment
  var closest = A.map(row => Math.min(...row))

  // median of all possible alignments
  const mid = Math.floor(closest.length / 2), nums = [...closest].sort((a, b) => a - b);
  return closest.length % 2 !== 0 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
};

function levenshtein(x, y, vocab, gamma = 0.0) {
  const n = x.length, m = y.length/vocab
  let cost = 0

  let A = new Array(n+1)
  for (let i = 0; i < n+1; i++) {
    A[i] = new Array(m+1)
    for (let j = 0; j < m+1; j++) {
      if (i === 0) {
        A[i][j] = j * gamma
      }
      else if (j === 0) {
        A[i][j] = i * gamma
      }
      else {
        cost = Math.log(1-y[vocab*(j-1)+x[i-1]])
        A[i][j] = Math.min(A[i-1][j]+gamma, A[i][j-1]+gamma, A[i-1][j-1]+cost);
      }
    }
  }

  return A[n][m]
};
