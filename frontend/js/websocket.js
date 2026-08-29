/**
 * websocket.js — WebSocket manager for progress + result streaming
 */

const WS_BASE = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/v1`;

export class VTHStockPilotWS {
  constructor() {
    this._ws = null;
    this._handlers = {};
  }

  connect(channel) {
    return new Promise((resolve, reject) => {
      // Map channel name to correct backend WebSocket URL
      // Backend routes:
      //   recommendations/{id} → /api/v1/recommendations/ws/{id}
      //   backtest/{id}        → /api/v1/ws/backtest/{id}
      let url;
      const recoMatch = channel.match(/^recommendations\/(.+)$/);
      const backtestMatch = channel.match(/^backtest\/(.+)$/);
      if (recoMatch) {
        url = `${WS_BASE}/recommendations/ws/${recoMatch[1]}`;
      } else if (backtestMatch) {
        url = `${WS_BASE}/ws/backtest/${backtestMatch[1]}`;
      } else {
        url = `${WS_BASE}/ws/${channel}`;
      }
      this._ws = new WebSocket(url);
      this._ws.onopen = () => resolve(this);
      this._ws.onerror = (e) => reject(e);
      this._ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          const handler = this._handlers[msg.type];
          if (handler) handler(msg);
        } catch { /* ignore malformed */ }
      };
      this._ws.onclose = () => {
        if (this._handlers.close) this._handlers.close();
      };
    });
  }

  send(data) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(data));
    }
  }

  on(type, handler) {
    this._handlers[type] = handler;
    return this;
  }

  close() {
    this._ws?.close();
    this._ws = null;
  }
}

/**
 * runWithProgress(channel, payload, { onProgress, onResult, onError })
 * High-level helper for recommendation / backtest WebSocket flows.
 */
export async function runWithProgress(channel, payload, { onProgress, onResult, onError }) {
  const ws = new VTHStockPilotWS();
  try {
    await ws.connect(channel);
    ws.on('progress', (msg) => onProgress?.(msg.stage, msg.percent, msg.meta));
    ws.on('result', (msg) => { onResult?.(msg.data); ws.close(); });
    ws.on('error', (msg) => { onError?.(msg.message); ws.close(); });
    ws.on('close', () => {});
    ws.send(payload);
  } catch (e) {
    onError?.(e.message);
  }
}
