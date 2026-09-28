const autocannon = require('autocannon');

const queries = [
  "what is machine learning",
  "what is the normal resting heart rate for adults",
  "what causes tides in the ocean",
  "how many chromosomes do humans have",
  "what is photosynthesis",
  "what is the capital of australia",
  "who invented the telephone",
  "what is the boiling point of water in fahrenheit",
  "what is the speed of light in vacuum",
  "how does penicillin work",
  "what is inflation in economics",
  "what is the function of red blood cells",
  "what is the chemical formula for water",
  "what is renewable energy",
  "what is the largest planet in our solar system"
];

let queryIndex = 0;

const instance = autocannon({
  url: 'http://127.0.0.1:8000',
  connections: 8,
  duration: 15,
  requests: queries.map(q => ({
    method: 'GET',
    path: `/search?q=${encodeURIComponent(q)}&mode=hybrid&top_k=5`
  }))
}, (err, result) => {
  if (err) {
    console.error('Autocannon test failed:', err);
    process.exit(1);
  }
  console.log('\n=== MULTI-QUERY LOAD TEST SUMMARY ===');
  console.log(`Total Requests Sent : ${result.requests.total}`);
  console.log(`Duration (seconds)  : ${result.duration}`);
  console.log(`Throughput (Req/Sec): ${result.requests.average}`);
  console.log(`Throughput (Bytes/s): ${(result.throughput.average / 1024).toFixed(2)} KB/s`);
  console.log(`2.5% Latency        : ${result.latency.p2_5} ms`);
  console.log(`50% (Median) Latency: ${result.latency.p50} ms`);
  console.log(`90% Latency         : ${result.latency.p90} ms`);
  console.log(`97.5% Latency       : ${result.latency.p97_5} ms`);
  console.log(`99% Latency (p99)   : ${result.latency.p99} ms`);
  console.log(`Max Latency         : ${result.latency.max} ms`);
  console.log(`2xx Responses       : ${result['2xx']}`);
  console.log(`Non-2xx Responses   : ${result.non2xx}`);
  console.log(`Errors / Timeouts   : ${result.errors} / ${result.timeouts}`);
  console.log(`SLA (<300ms p95)    : ${result.latency.p97_5 <= 300 ? 'PASSED [OK]' : 'FLAGGED'}`);
});

autocannon.track(instance, { renderProgressBar: false });
