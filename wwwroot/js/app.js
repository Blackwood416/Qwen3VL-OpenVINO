const conn = new signalR.HubConnectionBuilder()
            .withUrl("/chatHub")
            .withAutomaticReconnect()
            .build();
        conn.serverTimeoutInMilliseconds = 120000;
        let sessions = [], currSid = null, isGen = false, pending = null, currentMode = 'chat';
        let streamText = "", currBubble = null, currCursor = null;
        let localHistory = []; // { prompt, image, video, response }

        const defaultConfig = {
            Temperature: 0.7,
            TopP: 0.8,
            TopK: 20,
            MaxTokens: 512,
            RepetitionPenalty: 1.0,
            PresencePenalty: 1.5,
            FrequencyPenalty: 0.0,
            ContextWindow: 4096,
            Greedy: false,
            StopSequences: []
        };
        let currentConfig = { ...defaultConfig };

        function toggleSettings() {
            const side = document.getElementById('settings-sidebar');
            const over = document.getElementById('overlay');
            side.classList.toggle('open');
            over.classList.toggle('show');
        }

        let configSaveTimer = null;
        function updateCfg(key, val) {
            const map = { 
                temp: 'Temperature', topp: 'TopP', topk: 'TopK', 
                max: 'MaxTokens', rep: 'RepetitionPenalty',
                pres: 'PresencePenalty', freq: 'FrequencyPenalty',
                ctx: 'ContextWindow', greedy: 'Greedy', stop: 'StopSequences'
            };
            const field = map[key];
            if (key === 'greedy') {
                currentConfig[field] = val;
            } else if (key === 'stop') {
                currentConfig[field] = val.split(',').map(s => s.trim()).filter(s => s.length > 0);
            } else if (key === 'max' || key === 'topk' || key === 'ctx') {
                currentConfig[field] = parseInt(val);
            } else {
                currentConfig[field] = parseFloat(val);
            }
            const valDisplay = document.getElementById('val-' + key);
            if(valDisplay) valDisplay.innerText = val;
            // 防抖保存到服务器
            if (configSaveTimer) clearTimeout(configSaveTimer);
            configSaveTimer = setTimeout(() => {
                if (currSid) {
                    conn.invoke("SaveSessionConfig", currSid, JSON.stringify(currentConfig)).catch(e => console.error('Save config error:', e));
                }
            }, 500);
        }

        function applyConfigToUI(cfg) {
            currentConfig = { ...defaultConfig, ...cfg };
            const revMap = {
                Temperature: { id: 'temp', slider: true },
                TopP: { id: 'topp', slider: true },
                TopK: { id: 'topk', slider: true },
                MaxTokens: { id: 'max', slider: true },
                RepetitionPenalty: { id: 'rep', slider: true },
                PresencePenalty: { id: 'pres', slider: true },
                FrequencyPenalty: { id: 'freq', slider: true },
                ContextWindow: { id: 'ctx', slider: true },
                Greedy: { id: 'greedy', checkbox: true }
            };
            for (const [field, info] of Object.entries(revMap)) {
                const el = document.getElementById('cfg-' + info.id);
                const valEl = document.getElementById('val-' + info.id);
                if (!el) continue;
                if (info.checkbox) {
                    el.checked = !!currentConfig[field];
                } else {
                    el.value = currentConfig[field];
                }
                if (valEl) valEl.innerText = currentConfig[field];
            }
        }

        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('active'));
            const selectedCard = document.getElementById('card-' + mode);
            if (selectedCard) selectedCard.classList.add('active');
        }

        // Custom Marked Renderer to ensure safety and styles
        const renderer = new marked.Renderer();
        marked.setOptions({ renderer, gfm: true, breaks: true });

        // Helper: Render Markdown safely
        function renderMd(text) {
            try {
                return marked.parse(text || "");
            } catch (e) {
                console.error("Markdown execution failed:", e);
                return text;
            }
        }

        conn.on("ReceiveToken", t => {
            if (!isGen) return; // 忽略停止后仍在传输中的残余 token
            if(!currBubble) appendAssistantMsg("", false);
            const indicator = currBubble.parentElement.querySelector('.warmup-indicator');
            if (indicator) indicator.style.display = 'none';
            
            streamText += t;
            currBubble.innerHTML = renderMd(streamText + " ▌");
            scrollToEnd();
        });

        conn.on("ReceiveStatus", msg => {
            let indicator = null;
            if (currBubble) {
                indicator = currBubble.parentElement.querySelector('.warmup-indicator');
            } else {
                const indicators = document.querySelectorAll('.warmup-indicator');
                if (indicators.length > 0) indicator = indicators[indicators.length - 1];
            }

            if (indicator) {
                if (msg) {
                    indicator.style.display = 'flex';
                    indicator.querySelector('span').innerText = msg;
                } else {
                    indicator.style.display = 'none';
                }
            }
        });

        function resetStreamState() {
            currBubble = null;
            streamText = "";
        }

        conn.on("ReceiveStats", s => {
            if (!currBubble) return;
            let footer = currBubble.parentElement.querySelector('.bubble-footer');
            if (!footer) {
                footer = document.createElement('div');
                footer.className = 'bubble-footer';
                currBubble.parentElement.appendChild(footer);
            }
            footer.innerHTML = `
                <div class="stat-item"><i class="fas fa-bolt"></i> ${s.tokensPerSecond.toFixed(1)} t/s</div>
                <div class="stat-item"><i class="fas fa-microchip"></i> Prefill: ${(s.prefillMs/1000).toFixed(2)}s</div>
                <div class="stat-item"><i class="fas fa-keyboard"></i> Tokens: ${s.tokenCount}</div>
            `;
            // Removed syncSessions() here to fix Bug B (UI flickering/loss of sync on rapid token updates)
            updateUI();
        });

        conn.on("GenerationStarted", () => { 
            isGen = true; 
            streamText = ""; 
            updateUI(); 
            // 如果 currBubble 已经通过 retry 设置好了，就不再创建新气泡
            if (!currBubble) appendAssistantMsg("", false); 
        });
        conn.on("GenerationComplete", (turnIdx, versionIdx) => { 
            isGen = false; 
            updateUI(); 
            if (currBubble) {
                const matchTool = streamText.match(/<tool_call>(.*?)<\/tool_call>/s);
                const matchJson = streamText.match(/```json\s*(\[.*?\])\s*```/s);
                const matchPoint = streamText.match(/```json\s*(\{.*?"point_2d".*?\})\s*```/s);

                if (matchTool) {
                    try {
                        const action = JSON.parse(matchTool[1]);
                        if (action.arguments && action.arguments.coordinate) renderPoint(action.arguments.coordinate);
                    } catch(e) {}
                } else if (matchJson) {
                    try {
                        const boxes = JSON.parse(matchJson[1]);
                        if (Array.isArray(boxes) && boxes.length > 0 && boxes[0].bbox_2d) renderBoundingBoxes(boxes);
                    } catch(e) {}
                } else if (matchPoint) {
                    try {
                        const pt = JSON.parse(matchPoint[1]);
                        if (pt.point_2d) renderPoint(pt.point_2d);
                    } catch(e) {}
                }
                currBubble.innerHTML = renderMd(processToolCalls(streamText));
            }
            // Update local history
            const turn = localHistory.find((_, i) => i === turnIdx);
            if (turn) {
                if (!turn.versions) turn.versions = [];
                turn.versions[versionIdx] = streamText;
                turn.currentVersion = versionIdx;
            }
            resetStreamState();
            syncSessions();
            // Re-render this specific bubble to show version pager
            if (turn) renderVersionPager(turnIdx);
        });

        conn.on("Error", msg => {
            isGen = false;
            updateUI();
            alert("Error: " + msg);
            resetStreamState();
        });

        conn.on("GenerationCancelled", () => {
            isGen = false;
            updateUI();
            // stop() 已经处理了 UI 标记，这里只做清理
            if (currBubble) {
                currBubble.innerHTML = renderMd(streamText) + " <span style='color:var(--text-muted)'>(已停止)</span>";
                resetStreamState();
            }
        });

        async function start() {
            try { 
                await conn.start(); 
                document.getElementById('status').innerHTML = '<i class="fas fa-circle"></i> 就绪';
                document.getElementById('status').style.color = '#10b981';
                await syncSessions();
                if (sessions.length > 0) {
                    const first = typeof sessions[0] === 'object' ? sessions[0].id : sessions[0];
                    await switchSession(first);
                } else {
                    await createNewSession();
                }
            } catch (e) { 
                console.error("Connection failed", e);
                document.getElementById('status').innerHTML = '<i class="fas fa-exclamation-triangle"></i> 连接失败';
                document.getElementById('status').style.color = '#ef4444';
                updateUI(); 
                setTimeout(start, 5000); 
            }
        }




        async function syncSessions() {
            const saved = await conn.invoke("ListSessions");
            sessions = saved || [];
            renderSessions();
        }

        async function createNewSession() {
            if(isGen) return;
            const sid = await conn.invoke("CreateSession");
            currSid = sid;
            localHistory = [];
            applyConfigToUI(defaultConfig); // 重置为默认配置
            await syncSessions();
            document.getElementById('chat-wrapper').innerHTML = '';
            document.getElementById('welcome-screen').style.display = 'block';
            document.getElementById('chat-wrapper').style.display = 'none';
            resetStreamState();
            updateUI();
        }

        function renderSessions() {
            try {
                const listWrap = document.getElementById('session-list');
                if (!listWrap) return;
                
                let displayList = (Array.isArray(sessions) ? sessions : []).map(s => {
                    if (!s) return null;
                    if (typeof s === 'string') return { id: s, title: s };
                    return { id: s.id, title: s.title || s.id };
                }).filter(Boolean);
                
                listWrap.innerHTML = displayList.map(s => {
                    const sid = s.id || "";
                    const stitle = String(s.title || "未知会话").replace(/["']/g, '');
                    const isActive = sid === currSid ? 'active' : '';
                    
                    return `
                        <div class="session-item ${isActive}" onclick="switchSession('${sid}')">
                            <div class="session-title" title="${stitle}" style="pointer-events: none;">${stitle}</div>
                            <div class="session-actions">
                                <i class="fas fa-edit action-icon" onclick="event.stopPropagation(); renameSession('${sid}', '${stitle}')"></i>
                                <i class="fas fa-trash action-icon" onclick="event.stopPropagation(); deleteSession('${sid}')"></i>
                            </div>
                        </div>
                    `;
                }).join('');
            } catch (err) {
                console.error("renderSessions error:", err);
            }
        }

        async function renameSession(sid, oldTitle) {
            const newTitle = prompt("重命名会话", oldTitle);
            if (newTitle && newTitle !== oldTitle) {
                const ok = await conn.invoke("RenameSession", sid, newTitle);
                if (ok) await syncSessions();
            }
        }

        async function deleteSession(sid) {
            if (!confirm("确定要物理删除该会话及其所有存盘吗？此操作不可撤销。")) return;
            const ok = await conn.invoke("DeleteSession", sid);
            if (ok) {
                if (sid === currSid) await createNewSession();
                else await syncSessions();
            }
        }

        async function switchSession(sid) {
            if(isGen || sid === currSid) return;
            try {
                const history = await conn.invoke("SwitchSession", sid);
                currSid = sid;
                await syncSessions();
                
                localHistory = (history || []).map((item, idx) => ({
                    prompt: item.prompt,
                    versions: [item.response],
                    currentVersion: 0,
                    imageName: item.imagePath,
                    videoName: item.videoPath,
                    stats: item.stats ? JSON.parse(item.stats) : null
                }));
                renderHistory();

                // 加载该会话的 Generation Config
                const cfgJson = await conn.invoke("GetSessionConfig", sid);
                if (cfgJson) {
                    try { applyConfigToUI(JSON.parse(cfgJson)); } catch(e) {}
                } else {
                    applyConfigToUI(defaultConfig);
                }
                updateUI();
            } catch (err) {
                console.error("Failed to switch session:", err);
            }
        }

// Duplicate renderHistory(history) removed

        function appendMsg(role, text, upload, turnIdx) {
            const wrap = document.getElementById('chat-wrapper');
            const row = document.createElement('div');
            row.className = `message-row ${role}`;
            if (role === 'user' && turnIdx !== undefined) {
                row.setAttribute('data-user-turn-idx', turnIdx);
            }
            let att = "";
            if (upload && upload.fileType) {
                const url = upload.objectUrl || `/uploads/${upload.fileName}`;
                const imgClass = upload.fileType==='image' ? 'class="target-image"' : '';
                att = `<div class="attachment">${upload.fileType==='image'?`<img ${imgClass} src="${url}">`:`<video src="${url}" controls></video>`}</div>`;
            }
            let actions = '';
            if (role === 'user' && turnIdx !== undefined) {
                actions = `<div class="user-actions"><button class="action-btn" title="撤回" onclick="recallTurn(${turnIdx})"><i class="fas fa-undo"></i> 撤回</button></div>`;
            }
            row.innerHTML = `<div class="bubble"><div class="bubble-content">${att}${renderMd(text)}</div>${actions}</div>`;
            wrap.appendChild(row);
            scrollToEnd();
        }

        function renderHistory() {
            const wrap = document.getElementById('chat-wrapper');
            wrap.innerHTML = '';
            localHistory.forEach((h, i) => {
                appendMsg('user', h.prompt, (h.imageName||h.videoName)?{fileName:h.imageName||h.videoName, fileType:h.imageName?'image':'video'}:null, i);
                appendAssistantMsg(h.versions ? h.versions[h.currentVersion] : "", true, i);
            });
            if (localHistory.length > 0) {
                document.getElementById('welcome-screen').style.display = 'none';
                document.getElementById('chat-wrapper').style.display = 'flex';
            } else {
                document.getElementById('welcome-screen').style.display = 'flex';
                document.getElementById('chat-wrapper').style.display = 'none';
            }
        }

        function appendAssistantMsg(text, silent, idx) {
            const wrap = document.getElementById('chat-wrapper');
            const row = document.createElement('div');
            row.className = 'message-row assistant';
            
            const messageIdx = idx !== undefined ? idx : (localHistory.length - 1);
            row.setAttribute('data-turn-idx', messageIdx);

            row.innerHTML = `
                <div class="bubble">
                    <div class="warmup-indicator"><i class="fas fa-sync-alt"></i><span>正在恢复对话上下文...</span></div>
                    <div class="bubble-content"></div>
                    <div class="message-footer">
                        <div class="version-pager" style="display: none;">
                            <button class="pager-btn" onclick="switchVersion(${messageIdx}, -1)"><i class="fas fa-chevron-left"></i></button>
                            <span class="pager-text">1 / 1</span>
                            <button class="pager-btn" onclick="switchVersion(${messageIdx}, 1)"><i class="fas fa-chevron-right"></i></button>
                        </div>
                        <div class="message-actions">
                            <button class="action-btn" title="复制" onclick="copyMsg(this, ${messageIdx})"><i class="far fa-copy"></i> 复制</button>
                            <button class="action-btn" title="重试" onclick="retry(${messageIdx})"><i class="fas fa-redo"></i> 重试</button>
                        </div>
                    </div>
                </div>
            `;
            
            const content = row.querySelector('.bubble-content');
            if (silent) {
                content.innerHTML = renderMd(processToolCalls(text));
                // Restore visual attachments for history
                const matchTool = text.match(/<tool_call>(.*?)<\/tool_call>/s);
                const matchJson = text.match(/```json\s*(\[.*?\])\s*```/s);
                const matchPoint = text.match(/```json\s*(\{.*?"point_2d".*?\})\s*```/s);
                if (matchTool) {
                    try {
                        const action = JSON.parse(matchTool[1]);
                        if (action.arguments && action.arguments.coordinate) renderPoint(action.arguments.coordinate);
                    } catch(e) {}
                } else if (matchJson) {
                    try {
                        const boxes = JSON.parse(matchJson[1]);
                        if (Array.isArray(boxes) && boxes.length > 0 && boxes[0].bbox_2d) renderBoundingBoxes(boxes);
                    } catch(e) {}
                } else if (matchPoint) {
                    try {
                        const pt = JSON.parse(matchPoint[1]);
                        if (pt.point_2d) renderPoint(pt.point_2d);
                    } catch(e) {}
                }
                // 渲染持久化的 stats footer
                const turnData = localHistory[messageIdx];
                if (turnData && turnData.stats) {
                    const s = turnData.stats;
                    const footer = document.createElement('div');
                    footer.className = 'bubble-footer';
                    footer.innerHTML = `
                        <div class="stat-item"><i class="fas fa-bolt"></i> ${s.tokensPerSecond?.toFixed(1) || '?'} t/s</div>
                        <div class="stat-item"><i class="fas fa-microchip"></i> Prefill: ${s.prefillMs ? (s.prefillMs/1000).toFixed(2) + 's' : '?'}</div>
                        <div class="stat-item"><i class="fas fa-keyboard"></i> Tokens: ${s.tokenCount || '?'}</div>
                    `;
                    row.querySelector('.bubble').appendChild(footer);
                }
                loadTurnVersions(messageIdx);
            } else {
                streamText = "";
                currBubble = content;
            }
            wrap.appendChild(row);
            scrollToEnd();
        }

        async function loadTurnVersions(idx) {
            const turn = localHistory[idx];
            if (!turn) return;
            const versions = await conn.invoke("GetTurnVersions", currSid, idx);
            if (versions && versions.length > 1) {
                turn.versions = versions;
                turn.currentVersion = versions.length - 1;
                renderVersionPager(idx);
            }
        }

        function renderVersionPager(idx) {
            const turn = localHistory[idx];
            const row = document.querySelector(`.message-row[data-turn-idx="${idx}"]`);
            if (!turn || !row) return;
            
            const pager = row.querySelector('.version-pager');
            if (turn.versions && turn.versions.length > 1) {
                pager.style.display = 'flex';
                pager.querySelector('.pager-text').innerText = `${turn.currentVersion + 1} / ${turn.versions.length}`;
            } else {
                pager.style.display = 'none';
            }
            const content = row.querySelector('.bubble-content');
            if (turn.versions) {
                content.innerHTML = renderMd(processToolCalls(turn.versions[turn.currentVersion]));
            }
        }

        function switchVersion(idx, delta) {
            const turn = localHistory[idx];
            if (!turn || !turn.versions) return;
            const next = turn.currentVersion + delta;
            if (next >= 0 && next < turn.versions.length) {
                turn.currentVersion = next;
                renderVersionPager(idx);
            }
        }

        function processToolCalls(text) {
            const matchTool = text.match(/<tool_call>(.*?)<\/tool_call>/s);
            if (matchTool) {
                try {
                    const action = JSON.parse(matchTool[1]);
                    if (action.arguments && action.arguments.coordinate) {
                        let log = `\n> 📍 **执行操作**: \`${action.name}\`\n`;
                        log += `> 🖱️ **动作**: \`${action.arguments.action}\`\n`;
                        log += `> 🎯 **相对坐标**: \`[${action.arguments.coordinate.join(', ')}]\``;
                        return text.replace(matchTool[0], log);
                    }
                } catch(e) {}
            }
            return text;
        }

        async function copyMsg(btn, idx) {
            let text = "";
            const turn = localHistory[idx];
            if (isGen && idx === localHistory.length - 1) {
                text = streamText;
            } else if (turn && turn.versions) {
                text = turn.versions[turn.currentVersion];
            }
            if(!text) return;

            const copyToClipboard = async (str) => {
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    return await navigator.clipboard.writeText(str);
                } else {
                    const el = document.createElement('textarea');
                    el.value = str;
                    el.setAttribute('readonly', '');
                    el.style.position = 'absolute';
                    el.style.left = '-9999px';
                    document.body.appendChild(el);
                    el.select();
                    document.execCommand('copy');
                    document.body.removeChild(el);
                    return Promise.resolve();
                }
            };

            try {
                await copyToClipboard(text);
                const original = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check"></i> 已复制';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.innerHTML = original;
                    btn.classList.remove('copied');
                }, 2000);
            } catch(e) {
                console.error("Copy failed", e);
            }
        }

        async function retry(idx) {
            if (isGen || idx >= localHistory.length) return;
            const turn = localHistory[idx];
            
            const row = document.querySelector(`.message-row[data-turn-idx="${idx}"]`);
            if (!row) return;
            currBubble = row.querySelector('.bubble-content');
            currBubble.innerHTML = "";
            streamText = "";
            isGen = true;
            updateUI();
            
            const imgName = turn.imageName || null;
            const vidName = turn.videoName || null;
            const config = { ...currentConfig, Mode: currentMode };
            
            console.log('[retry] invoking SendMessage:', currSid, turn.prompt, imgName, vidName, idx);
            try {
                await conn.invoke("SendMessage", currSid, turn.prompt, imgName, vidName, 8, config, idx);
                console.log('[retry] invoke resolved');
            } catch(e) {
                console.error('[retry] invoke failed:', e);
                isGen = false;
                updateUI();
                resetStreamState();
            }
        }
        

        async function handleFile(input) {
            const file = input.files[0]; if(!file) return;
            const fd = new FormData(); fd.append('file', file);
            const resp = await fetch('/api/upload', { method: 'POST', body: fd });
            const data = await resp.json();
            pending = { ...data, objectUrl: URL.createObjectURL(file) };
            const strip = document.getElementById('preview-strip');
            strip.style.display = 'flex';
            strip.innerHTML = `<div class="preview-item">${data.fileType==='image'?`<img src="${pending.objectUrl}">`:`<video src="${pending.objectUrl}"></video>`}<div class="btn-remove" onclick="removePending()">×</div></div>`;
            updateUI();
        }

        function removePending() { pending = null; document.getElementById('preview-strip').style.display='none'; document.getElementById('file-input').value=''; updateUI(); }
        async function send() {
            if (!currSid) {
                const sendBtn = document.getElementById('send-btn');
                const originalHtml = sendBtn.innerHTML;
                sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                await createNewSession();
                sendBtn.innerHTML = originalHtml;
                if (!currSid) return;
            }
            const input = document.getElementById('user-input');
            const txt = input.value.trim(); if(!txt && !pending) return;
            const up = pending; 
            const currentTurnIdx = localHistory.length;
            appendMsg('user', txt, up, currentTurnIdx);
            input.value = ''; input.style.height = 'auto'; removePending();
            
            document.getElementById('welcome-screen').style.display = 'none';
            document.getElementById('chat-wrapper').style.display = 'flex';
            
            const config = {
                ...currentConfig,
                Mode: currentMode
            };
            
            localHistory.push({
                prompt: txt,
                imageName: up?.fileType === 'image' ? up.fileName : null,
                videoName: up?.fileType === 'video' ? up.fileName : null,
                versions: [""],
                currentVersion: 0
            });

            console.log('[send] invoking SendMessage:', currSid, txt, up?.fileName, config);
            try {
                await conn.invoke("SendMessage", currSid, txt, 
                    up?.fileType==='image' ? up.fileName : null, 
                    up?.fileType==='video' ? up.fileName : null, 
                    8, config, null);
                console.log('[send] invoke resolved');
            } catch(e) {
                console.error('[send] invoke failed:', e);
                isGen = false;
                updateUI();
                resetStreamState();
            }
        }
        
        function renderPoint(coords) {
            const imgs = document.querySelectorAll('.target-image');
            if(imgs.length === 0) return;
            const img = imgs[imgs.length - 1]; // get the latest image
            const xPercent = coords[0] / 10;
            const yPercent = coords[1] / 10;
            
            const dot = document.createElement('div');
            dot.className = 'action-dot';
            dot.style.left = `${xPercent}%`;
            dot.style.top = `${yPercent}%`;
            img.parentElement.appendChild(dot);
        }

        function renderBoundingBoxes(boxes) {
            const imgs = document.querySelectorAll('.target-image');
            if(imgs.length === 0) return;
            const img = imgs[imgs.length - 1]; // get the latest image
            const parent = img.parentElement;

            boxes.forEach(box => {
                const [xmin, ymin, xmax, ymax] = box.bbox_2d;
                const top = (ymin / 1000) * 100;
                const left = (xmin / 1000) * 100;
                const width = ((xmax - xmin) / 1000) * 100;
                const height = ((ymax - ymin) / 1000) * 100;

                const boxEl = document.createElement('div');
                boxEl.className = 'grounding-box';
                boxEl.style.top = `${top}%`;
                boxEl.style.left = `${left}%`;
                boxEl.style.width = `${width}%`;
                boxEl.style.height = `${height}%`;

                const labelEl = document.createElement('div');
                labelEl.className = 'grounding-label';
                labelEl.innerText = box.label || 'object';
                boxEl.appendChild(labelEl);

                parent.appendChild(boxEl);
            });
        }
        
        async function stop() { 
            isGen = false;
            updateUI();
            if (currBubble) {
                currBubble.innerHTML = renderMd(streamText) + " <span style='color:var(--text-muted)'>(已停止)</span>";
            }
            resetStreamState();
            try { await conn.invoke("CancelGeneration"); } catch(e) {}
        }

        async function recallTurn(idx) {
            if (isGen || !currSid) return;
            const turn = localHistory[idx];
            if (!turn) return;
            try {
                await conn.invoke("RecallTurn", currSid, idx);
                // 回填文本到输入框
                const input = document.getElementById('user-input');
                input.value = turn.prompt;
                input.style.height = 'auto';
                input.style.height = input.scrollHeight + 'px';
                input.focus();
                // 恢复附件预览
                const fileName = turn.imageName || turn.videoName;
                if (fileName) {
                    const fileType = turn.imageName ? 'image' : 'video';
                    pending = { fileName, fileType };
                    const strip = document.getElementById('preview-strip');
                    const url = `/uploads/${fileName}`;
                    strip.style.display = 'flex';
                    strip.innerHTML = `<div class="preview-item">${fileType==='image'?`<img src="${url}">`:`<video src="${url}"></video>`}<div class="btn-remove" onclick="removePending()">×</div></div>`;
                }
                // 截断本地历史并重新渲染
                localHistory.splice(idx);
                renderHistory();
                updateUI();
            } catch(e) {
                console.error('Recall failed:', e);
            }
        }

        function updateUI() {
            const sendBtn = document.getElementById('send-btn');
            const stopBtn = document.getElementById('stop-btn');
            const userInput = document.getElementById('user-input');
            if (userInput) userInput.disabled = isGen;
            sendBtn.style.display = isGen ? 'none' : 'flex';
            stopBtn.style.display = isGen ? 'flex' : 'none';
            sendBtn.style.opacity = (!currSid || isGen) ? '0.5' : '1';
            sendBtn.style.pointerEvents = (!currSid || isGen) ? 'none' : 'auto';
            // Highlight New Chat button if welcome screen is showing
            const welcomeScreen = document.getElementById('welcome-screen');
            const isNewChat = welcomeScreen && welcomeScreen.style.display !== 'none';
            document.querySelector('.new-chat-btn').classList.toggle('active', isNewChat);
            renderSessions();
        }
        function scrollToEnd() { const c = document.getElementById('chat-container'); c.scrollTop = c.scrollHeight; }
        document.getElementById('user-input').addEventListener('input', updateUI);
        document.getElementById('user-input').addEventListener('keydown', e => { if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
        start();
