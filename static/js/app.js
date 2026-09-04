const { createApp } = Vue;

createApp({
    data() {
        return {
            currentView: 'dashboard',
            activeProject: null,
            activeTab: 'media',
            
            projects: [],
            loadingProjects: false,
            
            projectImages: [],
            selectedFrames: [],
            projectClasses: [],
            trainedModels: [],
            
            // Canvas state
            currentImage: null,
            currentBoxes: [],
            selectedBoxIndex: -1,
            selectedClassId: 0,
            canvasMode: 'draw', // 'draw' or 'select'
            fabricCanvas: null,
            imageNaturalWidth: 1,
            imageNaturalHeight: 1,
            
            // YOLOE Auto-Annotation state
            aiAutoAnnotateMode: 'text', // 'text' or 'visual'
            aiAutoAnnotateModel: 'yoloe-26n-seg.pt',
            aiAutoAnnotateConf: 0.25,
            aiAutoAnnotateText: '',
            aiAutoAnnotateOverwrite: false,
            aiAutoAnnotateLoading: false,
            
            canvasScale: 1,
            isPanning: false,
            lastPosX: 0,
            lastPosY: 0,
            isSpaceDown: false,
            isDrawing: false,
            drawStartX: 0,
            drawStartY: 0,
            activeDrawRect: null,
            
            // Dataset generator
            trainRatio: 0.8,
            datasetStats: null,
            generatingDataset: false,
            
            // Training (YOLO26 ONLY)
            trainForm: {
                model_name: 'yolo26n.pt',
                epochs: 100, // Default to 100 as recommended!
                batch_size: 16,
                imgsz: 640,
                device: 'auto', // Auto-selects CUDA on Windows/Linux or MPS on Mac M1-M4!
                run_name: 'train_run'
            },
            trainStatus: {
                is_running: false,
                pid: null
            },
            trainProgress: {
                epoch: 0,
                total_epochs: 0,
                metrics: {}
            },
            terminalLogs: [],
            
            // Export
            exportForm: {
                weights_path: null,
                format: 'onnx'
            },
            exporting: false,
            lastExportResult: null,
            
            // Modals
            showCreateModal: false,
            createForm: { name: '', description: '' },
            
            showVideoModal: false,
            videoForm: { step_mode: 'nth_frame', step_value: 5 },
            pendingVideoFile: null,
            
            // Upload & Classes
            newClassName: '',
            newClassColor: '#3b82f6',
            isDragging: false,
            uploading: false,
            
            // WebSocket
            ws: null,
            wsConnected: false,
            
            // Toast
            toast: { show: false, message: '', type: 'success' },
            toastTimer: null
        };
    },
    computed: {
        hasSelectedBox() {
            return this.selectedBoxIndex >= 0 && this.selectedBoxIndex < this.currentBoxes.length;
        },
        trainingRecommendations() {
            const warnings = [];
            const imgCount = this.projectImages.length;
            const epochs = this.trainForm.epochs || 0;
            
            if (imgCount < 50) {
                warnings.push(`Мало кадров в проекте (${imgCount} шт.). Рекомендуется минимум 50 изображений для уверенного распознавания на видео.`);
            }
            if (epochs < 50) {
                warnings.push(`Установлено мало эпох (${epochs}). Рекомендуется минимум 50 эпох, чтобы модель не выдавала «no detection».`);
            }
            return warnings;
        }
    },
    mounted() {
        this.fetchProjects();
        this.connectWebSocket();
        
        window.addEventListener('keydown', this.onKeyDown);
        window.addEventListener('keyup', this.onKeyUp);
    },
    beforeUnmount() {
        if (this.ws) this.ws.close();
        window.removeEventListener('keydown', this.onKeyDown);
        window.removeEventListener('keyup', this.onKeyUp);
    },
    methods: {
        showToast(message, type = 'success') {
            this.toast.message = message;
            this.toast.type = type;
            this.toast.show = true;
            if (this.toastTimer) clearTimeout(this.toastTimer);
            this.toastTimer = setTimeout(() => {
                this.toast.show = false;
            }, 3500);
        },
        
        // --- REST API: Projects & ZIP Sharing ---
        async fetchProjects() {
            this.loadingProjects = true;
            try {
                const res = await fetch('/api/projects');
                if (res.ok) {
                    this.projects = await res.json();
                }
            } catch (e) {
                this.showToast('Ошибка загрузки проектов', 'error');
            } finally {
                this.loadingProjects = false;
            }
        },
        async createProjectSubmit() {
            if (!this.createForm.name) return;
            try {
                const res = await fetch('/api/projects', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.createForm)
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Ошибка создания проекта');
                }
                const proj = await res.json();
                this.showCreateModal = false;
                this.createForm.name = '';
                this.createForm.description = '';
                this.showToast(`Проект '${proj.name}' создан`);
                await this.fetchProjects();
                this.openProject(proj.name);
            } catch (e) {
                this.showToast(e.message, 'error');
            }
        },
        async confirmDeleteProject(name) {
            if (!confirm(`Удалить проект '${name}' и все его данные?`)) return;
            try {
                const res = await fetch(`/api/projects/${name}`, { method: 'DELETE' });
                if (res.ok) {
                    this.showToast(`Проект '${name}' удален`);
                    if (this.activeProject === name) {
                        this.backToDashboard();
                    }
                    this.fetchProjects();
                }
            } catch (e) {
                this.showToast('Ошибка удаления проекта', 'error');
            }
        },
        triggerZipImport() {
            if (this.$refs.zipInput) this.$refs.zipInput.click();
        },
        async onZipSelect(event) {
            const files = event.target.files;
            if (!files || files.length === 0) return;
            const file = files[0];
            const formData = new FormData();
            formData.append('file', file);
            
            this.loadingProjects = true;
            try {
                const res = await fetch('/api/projects/import_zip', {
                    method: 'POST',
                    body: formData
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Ошибка импорта');
                }
                const proj = await res.json();
                this.showToast(`Проект '${proj.name}' импортирован!`);
                await this.fetchProjects();
                this.openProject(proj.name);
            } catch (e) {
                this.showToast(e.message, 'error');
            } finally {
                this.loadingProjects = false;
                event.target.value = '';
            }
        },
        exportProjectZip(projName) {
            window.open(`/api/projects/${projName}/export_zip`, '_blank');
        },
        async openProject(name) {
            this.activeProject = name;
            this.currentView = 'studio';
            this.activeTab = 'media';
            this.selectedFrames = [];
            await this.fetchProjectData();
        },
        backToDashboard() {
            this.currentView = 'dashboard';
            this.activeProject = null;
            this.currentImage = null;
            this.currentBoxes = [];
            this.selectedFrames = [];
            this.fetchProjects();
        },
        switchTab(tab) {
            this.activeTab = tab;
            if (tab === 'canvas') {
                this.$nextTick(() => {
                    if (!this.fabricCanvas) {
                        this.initCanvas();
                    }
                    if (this.currentImage) {
                        this.openInCanvas(this.currentImage);
                    } else if (this.projectImages.length > 0) {
                        this.openInCanvas(this.projectImages[0]);
                    }
                    setTimeout(() => {
                        if (this.fabricCanvas && this.$refs.canvasEl) {
                            const wrapper = this.$refs.canvasEl.parentElement;
                            if (wrapper && wrapper.clientWidth > 0 && (wrapper.clientWidth !== this.fabricCanvas.width || wrapper.clientHeight !== this.fabricCanvas.height)) {
                                this.fabricCanvas.setWidth(wrapper.clientWidth);
                                this.fabricCanvas.setHeight(wrapper.clientHeight);
                                this.fitImageToCanvas();
                                this.renderBoxesOnCanvas();
                                this.fabricCanvas.renderAll();
                            }
                        }
                    }, 150);
                });
            } else if (tab === 'export') {
                this.fetchModels();
            }
        },
        async fetchProjectData() {
            if (!this.activeProject) return;
            try {
                const [imgRes, clsRes, modRes] = await Promise.all([
                    fetch(`/api/${this.activeProject}/images`),
                    fetch(`/api/${this.activeProject}/classes`),
                    fetch(`/api/${this.activeProject}/models`)
                ]);
                if (imgRes.ok) this.projectImages = await imgRes.json();
                if (clsRes.ok) {
                    this.projectClasses = await clsRes.json();
                    if (this.projectClasses.length > 0 && !this.selectedClassId) {
                        this.selectedClassId = this.projectClasses[0].id;
                    }
                }
                if (modRes.ok) this.trainedModels = await modRes.json();
            } catch (e) {
                console.error(e);
            }
        },

        // --- Module 3.1: File & Class Management ---
        async addClass() {
            if (!this.newClassName) return;
            const nextId = this.projectClasses.length > 0 ? Math.max(...this.projectClasses.map(c => c.id)) + 1 : 0;
            const newCls = { id: nextId, name: this.newClassName.trim(), color: this.newClassColor };
            const updated = [...this.projectClasses, newCls];
            try {
                const res = await fetch(`/api/${this.activeProject}/classes`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(updated)
                });
                if (res.ok) {
                    this.projectClasses = await res.json();
                    this.newClassName = '';
                    this.selectedClassId = newCls.id;
                    this.showToast(`Класс '${newCls.name}' добавлен`);
                }
            } catch (e) {
                this.showToast('Ошибка сохранения классов', 'error');
            }
        },
        async removeClass(id) {
            const updated = this.projectClasses.filter(c => c.id !== id);
            try {
                const res = await fetch(`/api/${this.activeProject}/classes`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(updated)
                });
                if (res.ok) {
                    this.projectClasses = await res.json();
                    if (this.selectedClassId === id && this.projectClasses.length > 0) {
                        this.selectedClassId = this.projectClasses[0].id;
                    }
                }
            } catch (e) {
                this.showToast('Ошибка удаления класса', 'error');
            }
        },
        getClassColor(classId) {
            const cls = this.projectClasses.find(c => c.id === classId);
            return cls ? cls.color : '#3b82f6';
        },

        // Explicit Photo & Video triggers
        triggerPhotoInput() {
            if (this.$refs.photoInput) this.$refs.photoInput.click();
        },
        triggerVideoInput() {
            if (this.$refs.videoInput) this.$refs.videoInput.click();
        },
        onPhotoSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.handlePhotosUpload(Array.from(files));
            }
        },
        onVideoSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.handleVideoUpload(files[0]);
            }
        },
        onDragOver() {
            this.isDragging = true;
        },
        onDragLeave() {
            this.isDragging = false;
        },
        onDropPhotos(event) {
            this.isDragging = false;
            const files = event.dataTransfer.files;
            if (files && files.length > 0) {
                this.handlePhotosUpload(Array.from(files));
            }
        },
        onDropVideo(event) {
            this.isDragging = false;
            const files = event.dataTransfer.files;
            if (files && files.length > 0) {
                this.handleVideoUpload(files[0]);
            }
        },
        async handlePhotosUpload(fileList) {
            for (const file of fileList) {
                await this.uploadSingleFile(file, 'nth_frame', 5);
            }
            await this.fetchProjectData();
        },
        handleVideoUpload(file) {
            this.pendingVideoFile = file;
            this.showVideoModal = true;
        },
        async submitVideoUpload() {
            if (!this.pendingVideoFile) return;
            this.showVideoModal = false;
            await this.uploadSingleFile(this.pendingVideoFile, this.videoForm.step_mode, this.videoForm.step_value);
            this.pendingVideoFile = null;
            await this.fetchProjectData();
        },
        cancelVideoUpload() {
            this.showVideoModal = false;
            this.pendingVideoFile = null;
        },
        async uploadSingleFile(file, stepMode, stepVal) {
            this.uploading = true;
            const formData = new FormData();
            formData.append('file', file);
            formData.append('step_mode', stepMode);
            formData.append('step_value', stepVal);
            
            try {
                const res = await fetch(`/api/${this.activeProject}/upload`, {
                    method: 'POST',
                    body: formData
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Ошибка загрузки');
                }
                const data = await res.json();
                if (data.type === 'video') {
                    this.showToast(`Извлечено кадров: ${data.count}`);
                } else {
                    this.showToast(`Загружено: ${file.name}`);
                }
            } catch (e) {
                this.showToast(`Ошибка: ${e.message}`, 'error');
            } finally {
                this.uploading = false;
            }
        },
        async deleteImage(filename) {
            if (!confirm(`Удалить ${filename}?`)) return;
            try {
                const res = await fetch(`/api/${this.activeProject}/images/${filename}`, { method: 'DELETE' });
                if (res.ok) {
                    this.projectImages = this.projectImages.filter(i => i.filename !== filename);
                    if (this.currentImage && this.currentImage.filename === filename) {
                        this.currentImage = null;
                        this.currentBoxes = [];
                        if (this.fabricCanvas) {
                            this.fabricCanvas.clear();
                            this.fabricCanvas.renderAll();
                        }
                    }
                    this.showToast(`Файл удален`);
                }
            } catch (e) {
                this.showToast('Ошибка удаления файла', 'error');
            }
        },
        toggleSelectAllFrames() {
            if (this.selectedFrames.length === this.projectImages.length && this.projectImages.length > 0) {
                this.selectedFrames = [];
            } else {
                this.selectedFrames = this.projectImages.map(i => i.filename);
            }
        },
        async deleteSelectedFrames() {
            if (this.selectedFrames.length === 0 || !this.activeProject) return;
            if (!confirm(`Удалить выбранные кадры (${this.selectedFrames.length} шт.)? Это действие необратимо!`)) return;
            try {
                const res = await fetch(`/api/${this.activeProject}/images/batch_delete`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filenames: this.selectedFrames })
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Ошибка удаления кадров');
                }
                const data = await res.json();
                this.showToast(`Удалено кадров: ${data.deleted_count}`);
                
                if (this.currentImage && this.selectedFrames.includes(this.currentImage.filename)) {
                    this.currentImage = null;
                    this.currentBoxes = [];
                    if (this.fabricCanvas) {
                        this.fabricCanvas.clear();
                        this.fabricCanvas.renderAll();
                    }
                }
                this.selectedFrames = [];
                await this.fetchProjectData();
            } catch (e) {
                this.showToast(e.message, 'error');
            }
        },
        async deleteUnannotatedFrames() {
            if (!this.activeProject) return;
            const unannotatedCount = this.projectImages.filter(i => !i.annotated).length;
            if (unannotatedCount === 0) {
                this.showToast('В проекте нет неразмеченных кадров (все кадры размечены)!');
                return;
            }
            if (!confirm(`Удалить ВСЕ неразмеченные кадры (${unannotatedCount} шт.) из проекта '${this.activeProject}'? Это действие необратимо!`)) return;
            try {
                const res = await fetch(`/api/${this.activeProject}/images/delete_unannotated`, {
                    method: 'POST'
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Ошибка удаления неразмеченных кадров');
                }
                const data = await res.json();
                this.showToast(`${data.message || 'Удалено кадров: ' + data.deleted_count}`);
                
                if (this.currentImage && !this.currentImage.annotated) {
                    this.currentImage = null;
                    this.currentBoxes = [];
                    if (this.fabricCanvas) {
                        this.fabricCanvas.clear();
                        this.fabricCanvas.renderAll();
                    }
                }
                this.selectedFrames = [];
                await this.fetchProjectData();
            } catch (e) {
                this.showToast(e.message, 'error');
            }
        },

        // --- Module 3.2: Fabric.js Annotation Canvas ---
        initCanvas() {
            if (this.fabricCanvas) return;
            const el = this.$refs.canvasEl;
            if (!el) return;
            
            const wrapper = el.parentElement;
            this.fabricCanvas = new fabric.Canvas('fabricCanvas', {
                width: wrapper.clientWidth || 800,
                height: wrapper.clientHeight || 600,
                selection: true,
                preserveObjectStacking: true,
                fireRightClick: true,
                stopContextMenu: true
            });
            window.addEventListener('resize', () => {
                if (this.fabricCanvas && this.$refs.canvasEl) {
                    const wrap = this.$refs.canvasEl.parentElement;
                    if (wrap && wrap.clientWidth > 0 && wrap.clientHeight > 0) {
                        this.fabricCanvas.setWidth(wrap.clientWidth);
                        this.fabricCanvas.setHeight(wrap.clientHeight);
                        this.fitImageToCanvas();
                        this.renderBoxesOnCanvas();
                        this.fabricCanvas.renderAll();
                    }
                }
            });
            
            this.fabricCanvas.on('mouse:down', (o) => {
                const evt = o.e;
                if (evt.altKey || evt.button === 2 || this.isSpaceDown) {
                    this.isPanning = true;
                    this.fabricCanvas.selection = false;
                    this.lastPosX = evt.clientX;
                    this.lastPosY = evt.clientY;
                    return;
                }
                
                if (this.canvasMode === 'draw' && !o.target) {
                    this.isDrawing = true;
                    this.fabricCanvas.selection = false;
                    const pointer = this.fabricCanvas.getPointer(evt);
                    this.drawStartX = pointer.x;
                    this.drawStartY = pointer.y;
                    
                    const color = this.getClassColor(this.selectedClassId);
                    this.activeDrawRect = new fabric.Rect({
                        left: this.drawStartX,
                        top: this.drawStartY,
                        width: 0,
                        height: 0,
                        fill: color + '33',
                        stroke: color,
                        strokeWidth: 2 / this.canvasScale,
                        selectable: false,
                        evented: false,
                        strokeUniform: true
                    });
                    this.fabricCanvas.add(this.activeDrawRect);
                }
            });
            
            this.fabricCanvas.on('mouse:move', (o) => {
                if (this.isPanning && o.e) {
                    const e = o.e;
                    const vpt = this.fabricCanvas.viewportTransform;
                    vpt[4] += e.clientX - self.lastPosX;
                    vpt[5] += e.clientY - self.lastPosY;
                    this.fabricCanvas.requestRenderAll();
                    self.lastPosX = e.clientX;
                    self.lastPosY = e.clientY;
                    return;
                }
                
                if (this.isDrawing && this.activeDrawRect) {
                    const pointer = this.fabricCanvas.getPointer(o.e);
                    if (pointer.x < this.drawStartX) {
                        this.activeDrawRect.set({ left: pointer.x });
                    }
                    if (pointer.y < this.drawStartY) {
                        this.activeDrawRect.set({ top: pointer.y });
                    }
                    this.activeDrawRect.set({
                        width: Math.abs(pointer.x - this.drawStartX),
                        height: Math.abs(pointer.y - this.drawStartY)
                    });
                    this.fabricCanvas.renderAll();
                }
            });
            
            this.fabricCanvas.on('mouse:up', (o) => {
                self = this;
                this.isPanning = false;
                this.fabricCanvas.selection = this.canvasMode === 'select';
                
                if (this.isDrawing && this.activeDrawRect) {
                    this.isDrawing = false;
                    const w = this.activeDrawRect.width;
                    const h = this.activeDrawRect.height;
                    
                    if (w > 5 && h > 5) {
                        const left = this.activeDrawRect.left;
                        const top = this.activeDrawRect.top;
                        
                        const newBox = {
                            class_id: this.selectedClassId,
                            x_min: left,
                            y_min: top,
                            width: w,
                            height: h
                        };
                        
                        this.currentBoxes.push(newBox);
                        this.saveAnnotations();
                    }
                    
                    this.fabricCanvas.remove(this.activeDrawRect);
                    this.activeDrawRect = null;
                    this.renderBoxesOnCanvas();
                }
            });
            
            this.fabricCanvas.on('mouse:wheel', (opt) => {
                const delta = opt.e.deltaY;
                let zoom = this.fabricCanvas.getZoom();
                zoom *= 0.999 ** delta;
                if (zoom > 10) zoom = 10;
                if (zoom < 0.1) zoom = 0.1;
                this.canvasScale = zoom;
                this.fabricCanvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
                opt.e.preventDefault();
                opt.e.stopPropagation();
            });
            
            this.fabricCanvas.on('object:modified', (e) => {
                const obj = e.target;
                if (obj && obj.boxIndex !== undefined) {
                    const idx = obj.boxIndex;
                    if (this.currentBoxes[idx]) {
                        this.currentBoxes[idx].x_min = obj.left;
                        this.currentBoxes[idx].y_min = obj.top;
                        this.currentBoxes[idx].width = obj.width * obj.scaleX;
                        this.currentBoxes[idx].height = obj.height * obj.scaleY;
                        this.saveAnnotations();
                    }
                }
            });
            
            this.fabricCanvas.on('selection:created', (e) => this.onCanvasSelection(e));
            this.fabricCanvas.on('selection:updated', (e) => this.onCanvasSelection(e));
            this.fabricCanvas.on('selection:cleared', () => {
                this.selectedBoxIndex = -1;
            });
        },
        onCanvasSelection(e) {
            const obj = e.selected ? e.selected[0] : null;
            if (obj && obj.boxIndex !== undefined) {
                this.selectedBoxIndex = obj.boxIndex;
            } else {
                this.selectedBoxIndex = -1;
            }
        },
        onKeyDown(e) {
            if (e.code === 'Space' && !this.isSpaceDown && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                this.isSpaceDown = true;
                if (this.fabricCanvas) {
                    this.fabricCanvas.defaultCursor = 'grab';
                    this.fabricCanvas.renderAll();
                }
            }
            if ((e.code === 'Delete' || e.code === 'Backspace') && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                if (this.currentView === 'studio' && this.activeTab === 'canvas' && this.hasSelectedBox) {
                    e.preventDefault();
                    this.deleteSelectedBox();
                }
            }
            // Arrow navigation between images
            if ((e.code === 'ArrowLeft' || e.code === 'KeyA') && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA' && document.activeElement.tagName !== 'SELECT') {
                if (this.currentView === 'studio' && this.activeTab === 'canvas') {
                    e.preventDefault();
                    this.navigateImage(-1);
                }
            }
            if ((e.code === 'ArrowRight' || e.code === 'KeyD') && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA' && document.activeElement.tagName !== 'SELECT') {
                if (this.currentView === 'studio' && this.activeTab === 'canvas') {
                    e.preventDefault();
                    this.navigateImage(1);
                }
            }
        },
        onKeyUp(e) {
            if (e.code === 'Space') {
                this.isSpaceDown = false;
                if (this.fabricCanvas) {
                    this.fabricCanvas.defaultCursor = 'default';
                    this.fabricCanvas.renderAll();
                }
            }
        },
        navigateImage(step) {
            if (!this.currentImage || this.projectImages.length === 0) return;
            const idx = this.projectImages.findIndex(i => i.filename === this.currentImage.filename);
            if (idx === -1) return;
            let nextIdx = idx + step;
            if (nextIdx < 0) nextIdx = this.projectImages.length - 1;
            if (nextIdx >= this.projectImages.length) nextIdx = 0;
            this.openInCanvas(this.projectImages[nextIdx]);
        },
        zoomCanvas(factor) {
            if (!this.fabricCanvas) return;
            let zoom = this.fabricCanvas.getZoom() * factor;
            if (zoom > 10) zoom = 10;
            if (zoom < 0.1) zoom = 0.1;
            this.canvasScale = zoom;
            this.fabricCanvas.setZoom(zoom);
            this.fabricCanvas.renderAll();
        },
        resetCanvasZoom() {
            if (!this.fabricCanvas || !this.currentImage) return;
            this.fabricCanvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
            this.canvasScale = 1;
            this.fitImageToCanvas();
        },
        async openInCanvas(img) {
            this.currentImage = img;
            if (this.activeTab !== 'canvas') {
                this.activeTab = 'canvas';
            }
            this.$nextTick(async () => {
                if (!this.fabricCanvas) this.initCanvas();
                await this.loadImageIntoCanvas(img);
            });
        },
        async loadImageIntoCanvas(img) {
            this.fabricCanvas.clear();
            
            try {
                const res = await fetch(`/api/${this.activeProject}/annotations`);
                if (res.ok) {
                    const allAnn = await res.json();
                    this.currentBoxes = allAnn[img.filename] || [];
                } else {
                    this.currentBoxes = [];
                }
            } catch (e) {
                this.currentBoxes = [];
            }
            
            fabric.Image.fromURL(img.url, (fImg) => {
                if (!this.fabricCanvas) return;
                const wrapper = this.$refs.canvasEl ? this.$refs.canvasEl.parentElement : null;
                if (wrapper && wrapper.clientWidth > 0 && wrapper.clientHeight > 0) {
                    if (wrapper.clientWidth !== this.fabricCanvas.width || wrapper.clientHeight !== this.fabricCanvas.height) {
                        this.fabricCanvas.setWidth(wrapper.clientWidth);
                        this.fabricCanvas.setHeight(wrapper.clientHeight);
                    }
                }
                
                this.imageNaturalWidth = fImg.width || 1;
                this.imageNaturalHeight = fImg.height || 1;
                
                fImg.set({
                    left: 0,
                    top: 0,
                    selectable: false,
                    evented: false
                });
                
                this.fabricCanvas.setBackgroundImage(fImg, () => {
                    this.fitImageToCanvas();
                    this.renderBoxesOnCanvas();
                    this.fabricCanvas.renderAll();
                });
            }, { crossOrigin: 'anonymous' });
        },
        fitImageToCanvas() {
            if (!this.fabricCanvas || !this.imageNaturalWidth) return;
            const cw = this.fabricCanvas.width;
            const ch = this.fabricCanvas.height;
            
            const scaleX = cw / this.imageNaturalWidth;
            const scaleY = ch / this.imageNaturalHeight;
            const scale = Math.min(scaleX, scaleY) * 0.95;
            
            this.canvasScale = scale;
            this.fabricCanvas.setZoom(scale);
            
            const offsetX = (cw - this.imageNaturalWidth * scale) / 2;
            const offsetY = (ch - this.imageNaturalHeight * scale) / 2;
            this.fabricCanvas.setViewportTransform([scale, 0, 0, scale, offsetX, offsetY]);
            this.fabricCanvas.renderAll();
        },
        renderBoxesOnCanvas() {
            if (!this.fabricCanvas) return;
            const toRemove = this.fabricCanvas.getObjects().filter(o => o.isBox);
            toRemove.forEach(o => this.fabricCanvas.remove(o));
            
            this.currentBoxes.forEach((box, idx) => {
                const color = this.getClassColor(box.class_id);
                const rect = new fabric.Rect({
                    left: box.x_min,
                    top: box.y_min,
                    width: box.width,
                    height: box.height,
                    fill: color + '33',
                    stroke: color,
                    strokeWidth: 2 / this.canvasScale,
                    strokeUniform: true,
                    selectable: this.canvasMode === 'select',
                    hasRotatingPoint: false,
                    cornerColor: color,
                    cornerSize: 8 / this.canvasScale,
                    transparentCorners: false,
                    isBox: true,
                    boxIndex: idx
                });
                this.fabricCanvas.add(rect);
            });
            this.fabricCanvas.renderAll();
        },
        selectBoxByIndex(idx) {
            this.selectedBoxIndex = idx;
            if (this.fabricCanvas && this.canvasMode === 'select') {
                const objs = this.fabricCanvas.getObjects().filter(o => o.isBox && o.boxIndex === idx);
                if (objs.length > 0) {
                    this.fabricCanvas.setActiveObject(objs[0]);
                    this.fabricCanvas.renderAll();
                }
            }
        },
        onBoxClassChange(idx, newClsId) {
            if (this.currentBoxes[idx]) {
                this.currentBoxes[idx].class_id = newClsId;
                this.renderBoxesOnCanvas();
                this.saveAnnotations();
            }
        },
        deleteSelectedBox() {
            if (this.hasSelectedBox) {
                this.deleteBoxByIndex(this.selectedBoxIndex);
            }
        },
        deleteBoxByIndex(idx) {
            this.currentBoxes.splice(idx, 1);
            this.selectedBoxIndex = -1;
            this.renderBoxesOnCanvas();
            this.saveAnnotations();
        },
        async saveAnnotations() {
            if (!this.currentImage || !this.activeProject) return;
            try {
                const res = await fetch(`/api/${this.activeProject}/annotations`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filename: this.currentImage.filename,
                        boxes: this.currentBoxes,
                        image_width: this.imageNaturalWidth,
                        image_height: this.imageNaturalHeight
                    })
                });
                if (res.ok) {
                    const img = this.projectImages.find(i => i.filename === this.currentImage.filename);
                    if (img) {
                        img.annotated = this.currentBoxes.length > 0;
                        img.boxes_count = this.currentBoxes.length;
                    }
                }
            } catch (e) {
                console.error("Autosave error:", e);
            }
        },

        fillPromptsWithProjectClasses() {
            this.aiAutoAnnotateText = this.projectClasses.map(c => c.name).join(', ');
        },
        async runAiAutoAnnotate() {
            if (!this.activeProject) return;
            if (this.aiAutoAnnotateMode === 'text' && !this.aiAutoAnnotateText.trim() && this.projectClasses.length === 0) {
                this.showToast('Укажите слова или классы для текстового поиска', 'error');
                return;
            }
            if (this.aiAutoAnnotateMode === 'visual' && !this.currentImage) {
                this.showToast('Откройте кадр-референс на холсте для визуальной разметки', 'error');
                return;
            }
            if (this.aiAutoAnnotateMode === 'visual' && (!this.currentBoxes || this.currentBoxes.length === 0)) {
                this.showToast('На текущем кадре-референсе нет размеченных рамок! Разметьте хотя бы 1 объект.', 'error');
                return;
            }

            if (this.currentImage) {
                await this.saveAnnotations();
            }

            this.aiAutoAnnotateLoading = true;
            this.showToast(`Запуск YOLOE авторазметки (${this.aiAutoAnnotateMode === 'text' ? 'Zero-Shot' : 'Few-Shot'})... Ожидайте!`);

            try {
                let prompts = [];
                if (this.aiAutoAnnotateMode === 'text') {
                    prompts = this.aiAutoAnnotateText.split(',').map(s => s.trim()).filter(s => s);
                    if (prompts.length === 0) {
                        prompts = this.projectClasses.map(c => c.name);
                    }
                }

                const payload = {
                    mode: this.aiAutoAnnotateMode,
                    model_name: this.aiAutoAnnotateModel,
                    conf_threshold: parseFloat(this.aiAutoAnnotateConf) || 0.25,
                    text_prompts: prompts,
                    reference_image: this.aiAutoAnnotateMode === 'visual' ? (this.currentImage ? this.currentImage.filename : '') : '',
                    overwrite: this.aiAutoAnnotateOverwrite
                };

                const res = await fetch(`/api/${this.activeProject}/auto_annotate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Ошибка авторазметки YOLOE');
                }

                const result = await res.json();
                this.showToast(`Готово! Размечено кадров: ${result.annotated_count}, добавлено рамок: ${result.total_boxes}`);
                
                await this.fetchProjectData();
                if (this.currentImage) {
                    const updatedImg = this.projectImages.find(i => i.filename === this.currentImage.filename);
                    if (updatedImg) this.currentImage = updatedImg;
                    await this.loadImageIntoCanvas(this.currentImage);
                } else if (this.projectImages.length > 0) {
                    await this.openInCanvas(this.projectImages[0]);
                }
            } catch (e) {
                this.showToast(e.message, 'error');
            } finally {
                this.aiAutoAnnotateLoading = false;
            }
        },

        // --- Module 3.3: Dataset Generator ---
        async generateDataset() {
            if (!this.activeProject) return;
            this.generatingDataset = true;
            this.datasetStats = null;
            try {
                const res = await fetch(`/api/${this.activeProject}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ train_ratio: this.trainRatio })
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Ошибка генерации');
                }
                this.datasetStats = await res.json();
                this.showToast('Датасет сформирован!');
                await this.fetchProjects();
            } catch (e) {
                this.showToast(e.message, 'error');
            } finally {
                this.generatingDataset = false;
            }
        },

        // --- Module 3.4: Training Studio (YOLO26 ONLY) ---
        async startTrain() {
            if (!this.activeProject) return;
            try {
                const res = await fetch(`/api/${this.activeProject}/train`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.trainForm)
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Ошибка старта обучения');
                }
                const data = await res.json();
                this.trainStatus = { is_running: true, pid: data.pid };
                this.trainProgress = { epoch: 0, total_epochs: this.trainForm.epochs, metrics: {} };
                this.showToast(`Обучение запущено (PID: ${data.pid})`);
            } catch (e) {
                this.showToast(e.message, 'error');
            }
        },
        async stopTrain() {
            if (!this.activeProject) return;
            try {
                const res = await fetch(`/api/${this.activeProject}/train/stop`, { method: 'POST' });
                if (res.ok) {
                    const data = await res.json();
                    this.trainStatus.is_running = false;
                    this.showToast(data.message || 'Обучение остановлено.');
                }
            } catch (e) {
                this.showToast('Ошибка остановки', 'error');
            }
        },

        // --- Module 3.5: Export & Models ---
        async fetchModels() {
            if (!this.activeProject) return;
            try {
                const res = await fetch(`/api/${this.activeProject}/models`);
                if (res.ok) {
                    this.trainedModels = await res.json();
                }
            } catch (e) {
                console.error(e);
            }
        },
        async exportModelRun() {
            if (!this.activeProject) return;
            this.exporting = true;
            this.lastExportResult = null;
            try {
                const res = await fetch(`/api/${this.activeProject}/export`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.exportForm)
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Ошибка экспорта');
                }
                this.lastExportResult = await res.json();
                this.showToast(`Модель экспортирована в ${this.lastExportResult.format.toUpperCase()}`);
                await this.fetchModels();
            } catch (e) {
                this.showToast(e.message, 'error');
            } finally {
                this.exporting = false;
            }
        },

        // --- WebSockets ---
        connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/logs`;
            
            this.ws = new WebSocket(wsUrl);
            this.ws.onopen = () => {
                this.wsConnected = true;
            };
            this.ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'log' || msg.type === 'error') {
                        this.terminalLogs.push(msg);
                        this.$nextTick(() => {
                            const el = this.$refs.terminalScrollEl;
                            if (el) el.scrollTop = el.scrollHeight;
                        });
                    } else if (msg.type === 'progress') {
                        this.trainProgress.epoch = msg.epoch;
                        this.trainProgress.total_epochs = msg.total_epochs;
                        this.trainProgress.metrics = msg.metrics || {};
                    } else if (msg.type === 'status') {
                        if (msg.status === 'running') {
                            this.trainStatus.is_running = true;
                        } else if (msg.status === 'stopped' || msg.status === 'error') {
                            this.trainStatus.is_running = false;
                        }
                    } else if (msg.type === 'finish') {
                        this.trainStatus.is_running = false;
                        this.showToast(msg.message || 'Обучение завершено!');
                        this.fetchModels();
                    }
                } catch (e) {
                    console.error("WS Parse error", e);
                }
            };
            this.ws.onclose = () => {
                this.wsConnected = false;
                setTimeout(() => {
                    this.connectWebSocket();
                }, 3000);
            };
            this.ws.onerror = () => {
                this.wsConnected = false;
            };
        }
    }
}).mount('#app');
