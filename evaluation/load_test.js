import http from 'k6/http'
import { check } from 'k6'
import { Trend } from 'k6/metrics'

const baseUrl = __ENV.BASE_URL || 'http://host.docker.internal:8000'
const virtualUsers = Number(__ENV.VUS || 1)
const serverLatency = new Trend('server_latency_ms')

// One request per VU makes every iteration a concurrent user query, instead
// of repeatedly hitting the prototype's 20 requests/minute/IP safety limit.
export const options = {
  scenarios: {
    concurrent_chat_queries: {
      executor: 'per-vu-iterations',
      vus: virtualUsers,
      iterations: 1,
      maxDuration: '3m',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<60000'],
  },
}

export default function () {
  const response = http.post(
    `${baseUrl}/api/v1/chat/query`,
    JSON.stringify({
      question: 'Mức phạt vi phạm trong hợp đồng mua bán hàng hóa tối đa là bao nhiêu?',
      as_of_date: '2026-09-21',
    }),
    { headers: { 'Content-Type': 'application/json' }, timeout: '180s' },
  )
  const isJson = response.headers['Content-Type']?.includes('application/json')
  const body = isJson ? response.json() : null
  if (body?.latency?.total_ms !== undefined) serverLatency.add(body.latency.total_ms)
  check(response, {
    'HTTP 200': (result) => result.status === 200,
    'response is a safe chat result': () => body?.status === 'grounded' || body?.status === 'abstained',
  })
}
