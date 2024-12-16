import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.AudioRecorder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AudioRecorder") {
            // Hide trigger widget
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.setSize([250, 160]);
                this.isRecording = false;
                
                // Hide the trigger widget
                const triggerWidget = this.widgets.find(w => w.name === "trigger");
                if (triggerWidget) {
                    triggerWidget.type = "hidden";
                    triggerWidget.hidden = true;
                }
                
                return r;
            };

            nodeType.prototype.onDrawForeground = function(ctx) {
                if (!this.buttonRect) {
                    const [w, h] = this.size;
                    const margin = 10;
                    const buttonHeight = 30;
                    this.buttonRect = [margin, h - buttonHeight - margin, w - margin * 2, buttonHeight];
                }

                const [x, y, w, h] = this.buttonRect;

                ctx.fillStyle = this.isRecording ? "#ff4444" : "#2a2a2a";
                ctx.strokeStyle = "#666";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.roundRect(x, y, w, h, 4);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = "#fff";
                ctx.font = "14px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(
                    this.isRecording ? "Recording..." : "Start Recording",
                    x + w/2,
                    y + h/2
                );
            };

            nodeType.prototype.onMouseDown = function(event, local_pos) {
                if (!this.buttonRect) return false;

                const [x, y, w, h] = this.buttonRect;
                if (local_pos[0] >= x && local_pos[0] <= x + w &&
                    local_pos[1] >= y && local_pos[1] <= y + h) {
                    
                    if (!this.isRecording) {
                        this.isRecording = true;
                        
                        // Update trigger value
                        const triggerWidget = this.widgets.find(w => w.name === "trigger");
                        if (triggerWidget) {
                            triggerWidget.value = (triggerWidget.value || 0) + 1;
                        }
                        
                        // Force node to be dirty and queue prompt
                        this.setOutputData(0, undefined);
                        app.graph.setDirtyCanvas(true);
                        app.queuePrompt();

                        const duration = this.widgets.find(w => w.name === "duration").value;
                        setTimeout(() => {
                            this.isRecording = false;
                            app.graph.setDirtyCanvas(true);
                        }, (duration * 1000) + 500);
                    }
                    return true;
                }
                return false;
            };
        }
    }
});
