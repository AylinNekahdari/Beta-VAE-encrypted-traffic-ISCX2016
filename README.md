<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>β_CVAE_encrypted_traffic_ISCX2016 — README (Simple)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg:#0b0f14;--panel:#0f1620;--text:#e7eef7;--muted:#9fb0c4;
      --border:#1f2a37;--shadow:0 10px 30px rgba(0,0,0,.35);
      --radius:16px;--accent:#60a5fa;--ok:#34d399;--warn:#fbbf24;
    }
    @media (prefers-color-scheme: light){
      :root{--bg:#f8fafc;--panel:#ffffff;--text:#0f1720;--muted:#4b5563;--border:#e5e7eb;--shadow:0 8px 24px rgba(0,0,0,.08)}
    }
    *{box-sizing:border-box}
    html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,Arial,Helvetica,sans-serif;line-height:1.7}
    .wrap{max-width:900px;margin:40px auto;padding:0 20px}
    .card{background:linear-gradient(180deg,rgba(255,255,255,.02),transparent);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:22px}
    h1{margin:0 0 8px;font-weight:800;font-size:clamp(28px,4vw,40px)}
    p.lead{color:var(--muted);margin:0 0 8px}
    .figs{display:grid;grid-template-columns:1fr;gap:16px;margin:16px 0}
    @media(min-width:860px){.figs{grid-template-columns:1fr 1fr}}
    img{width:100%;height:auto;border-radius:12px;border:1px solid var(--border);box-shadow:var(--shadow)}
    .caption{font-size:12px;color:var(--muted);margin-top:6px}
    table{width:100%;border-collapse:separate;border-spacing:0;border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-top:6px}
    th,td{padding:10px 12px;border-bottom:1px solid var(--border);text-align:left}
    thead th{background:rgba(255,255,255,.04);font-weight:600}
    tbody tr:last-child td{border-bottom:0}
    .pill{display:inline-block;padding:2px 8px;border:1px solid var(--border);border-radius:999px;font-size:12px;color:var(--muted);margin-left:8px}
    .ok{color:var(--ok);font-weight:700}
    .warn{color:var(--warn);font-weight:700}
    footer{color:var(--muted);font-size:12px;margin-top:18px}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="card">
      <h1>β_CVAE_encrypted_traffic_ISCX2016</h1>
      <p class="lead">A simple β‑VAE pipeline on ISCX VPN‑nonVPN traffic: learn a compact latent space, visualize it with t‑SNE, then classify.</p>
    </header>

    <section class="card" style="margin-top:18px">
      <h2 style="margin:0 0 10px">What this does <span class="pill">short &amp; simple</span></h2>
      <p>
        We train a β‑VAE to compress network flow features into a low‑dimensional latent vector. 
        We show (1) training loss, (2) a t‑SNE view of the latent space, and (3) a compact summary of
        <em>overall</em> Accuracy / Precision / Recall / F1 for two classifiers trained on the latents.
      </p>
    </section>

    <section class="card" style="margin-top:18px">
      <h2 style="margin:0 0 10px">Plots</h2>
      <div class="figs">
        <figure>
          <img src="assets/loss.png" alt="Training & validation loss">
          <figcaption class="caption">β‑VAE training &amp; validation loss per epoch.</figcaption>
        </figure>
        <figure>
          <img src="assets/tsne.png" alt="t-SNE of latent space">
          <figcaption class="caption">t‑SNE projection of the learned latent space.</figcaption>
        </figure>
      </div>
    </section>

    <section class="card" style="margin-top:18px">
      <h2 style="margin:0 0 10px">Classification (overall metrics)</h2>
      <p class="caption" style="margin:0 0 8px">Reported as <strong>overall</strong> metrics (no per‑class tables).</p>
      <table>
        <thead>
          <tr>
            <th>Classifier on Latents</th>
            <th>Accuracy</th>
            <th>Precision (macro)</th>
            <th>Recall (macro)</th>
            <th>F1 (macro)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>RandomForest</td>
            <td class="ok">1.00</td>
            <td class="ok">1.00</td>
            <td class="ok">1.00</td>
            <td class="ok">1.00</td>
          </tr>
          <tr>
            <td>Logistic Regression</td>
            <td class="warn">0.33</td>
            <td class="warn">0.05</td>
            <td class="warn">0.14</td>
            <td class="warn">0.07</td>
          </tr>
        </tbody>
      </table>
      <footer>Notes: sklearn warned about undefined precision for some classes in Logistic Regression due to no predicted samples. RandomForest captures non‑linear structure in the latent space.</footer>
    </section>
  </div>
</body>
</html>
