document.addEventListener("DOMContentLoaded", async function() {
    const loadingScreen = document.getElementById('loadingScreen');
    Cesium.Ion.defaultAccessToken = CONFIG.ACCESSTOKEN;
    
    const oauth2Token = Cesium.Ion.defaultAccessToken;
    const baseUrl = 'https://api.cesium.com/v1/assets';

    async function fetchLatestAsset() {
        const params = new URLSearchParams({
            sortBy: 'DATE_ADDED',
            sortOrder: 'DESC',
            status: 'COMPLETE'
        });

        const response = await fetch(`${baseUrl}?${params.toString()}`, {
            headers: {
                'Authorization': `Bearer ${oauth2Token}`
            }
        });

        if (!response.ok) {
            throw new Error(`Error fetching assets: ${response.statusText}`);
        }

        const data = await response.json();
        return data.items[0];
    }   

    const viewer = new Cesium.Viewer("cesiumContainer", {
        shouldAnimate: true,
        animation: false,
        timeline: false,
        geocoder: false,
        sceneModePicker: false,
        baseLayerPicker: false,
        navigationHelpButton: false,
        homeButton: false
    });

    viewer.scene.globe.enableLighting = true;
    viewer.scene.sun = new Cesium.Sun();
    viewer.scene.moon = new Cesium.Moon();
    wireViewerResize(viewer);
    wireTrackpadPinchZoom(viewer);
    const topBottomInfoBox = document.getElementById('topBottomInfoBox');
    const clusterMemberListBox = document.getElementById('clusterMemberListBox');
    const modelModePanel = document.getElementById('modelModePanel');
    document.body.classList.add('orbx-mode-clusters');

    // on load or refresh, clear the search bars
    document.getElementById('searchInput').value = '';
    const clusterSearchInputEl = document.getElementById('clusterSearchInput');
    if (clusterSearchInputEl) {
        clusterSearchInputEl.value = '';
    }

    let dataSource;
    let highlightedEntities = [];
    let visibleOrbitEntities = [];
    const clusterLabelToEntities = new Map();
    const modeViewState = {
        unique: { initialized: false },
        clusters: { initialized: false },
    };
    let activeModelMode = 'clusters';
    let currentClusterLabel = null;
    let currentClusterHighlightId = null;
    // Guards against overlapping async cluster switches leaving old rings visible.
    let clusterDisplayGeneration = 0;
    // True while unique mode is showing a search/random neighbour set
    // rather than the default top/bottom uniqueness view for a regime.
    let uniqueModeShowingNeighbours = false;

    function delay(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }
    try {
        const latestAsset = await fetchLatestAsset();
        const assetId = latestAsset.id;
        
        const resource = await Cesium.IonResource.fromAssetId(assetId);
        dataSource = await Cesium.CzmlDataSource.load(resource);
        await viewer.dataSources.add(dataSource);
        syncClockToDataAvailability();
        // Fall back only if CZML had no availability window.
        if (!dataSource.entities.values.some((e) => e.availability && e.availability.length)) {
            viewer.clock.currentTime = Cesium.JulianDate.now();
        }
        viewer.clock.multiplier = 1;

        const step = 10;

        if (viewer.animation) {
            const animationViewModel = viewer.animation.viewModel;
            animationViewModel.playForwardViewModel.command.beforeExecute.addEventListener(function(commandInfo) {
                viewer.clock.multiplier += step;
            });

            animationViewModel.playReverseViewModel.command.beforeExecute.addEventListener(function(commandInfo) {
                viewer.clock.multiplier -= step;
            });
        }

        wireModelModeRadios();
        viewer.resize();

        const urlParams = new URLSearchParams(window.location.search);
        const idFromURL = urlParams.get('id');
        if (idFromURL) {
            performSearch(idFromURL);
        }

        dataSource.entities.values.forEach(entity => {
            entity.show = false;
            // Orbits-only view: never keep live trails or point markers.
            if (!String(entity.id || '').endsWith('-orbit-ring')) {
                entity.path = undefined;
                entity.point = undefined;
            }
        });
        rebuildClusterIndex();

    } catch (error) {
        console.log(error);
    }

    const infoBox = document.getElementById("infoBox");

    // In showCompressedInfo, update infoBox styling so its width fits content and text is smaller.
    function showCompressedInfo(entityData, mousePosition) {
        // Extract the entity id from the passed object or use the id directly.
        const entityId = (typeof entityData === 'object' && entityData.id) ? entityData.id : entityData;
        
        // Retrieve the entity from dataSource.
        const entity = dataSource && dataSource.entities && dataSource.entities.getById
            ? dataSource.entities.getById(entityId)
            : null;
        
        const now = Cesium.JulianDate.now();
        const offset = 10;
    
        // -----
            
        if (entity) {
            if (
                document.body.classList.contains('orbx-mode-unique') &&
                isSyntheticEntity(entity, now)
            ) {
                hideCompressedInfo();
                return;
            }
            const uniquenessStr = formatUniquenessScore(entity, now);
            let detailLines = '';
            if (document.body.classList.contains('orbx-mode-clusters')) {
                detailLines = `<div><strong>Role:</strong> ${getClusterRoleLabel(entity, now)}</div>`;
                const cl = getClusterLabelFromEntity(entity, now);
                if (cl !== null && cl !== -1) {
                    detailLines += `<div><strong>Cluster ID:</strong> ${cl}</div>`;
                }
            } else {
                detailLines = `<div><strong>Uniqueness:</strong> ${uniquenessStr}</div>`;
            }
            const displayNorad = formatHoverNoradId(entity, now);
            infoBox.innerHTML = `<div class="infoBox-hover-content">
                    <div><strong>NORAD ID:</strong> ${displayNorad}</div>
                    <div><strong>Name:</strong> ${entity.name || "N/A"}</div>
                    ${detailLines}
                </div>`;
        } else {
            infoBox.innerHTML = `<div class="infoBox-hover-content">Entity ID: ${entityId}</div>`;
        }
        
        infoBox.classList.add('infoBox--hover');
        infoBox.style.display = 'inline-block';
        infoBox.style.position = 'absolute';
        infoBox.style.zIndex = '9999';
    
        // Initially position the infoBox to the right and below the cursor.
        infoBox.style.left = (mousePosition.x + offset) + 'px';
        infoBox.style.top = (mousePosition.y + offset) + 'px';
    
        // After rendering, adjust position if the box overflows the viewport.
        const boxRect = infoBox.getBoundingClientRect();
    
        // Adjust horizontal position if overflowing right edge.
        if (boxRect.right > window.innerWidth) {
            infoBox.style.left = (mousePosition.x - boxRect.width - offset) + 'px';
        }
    
        // Adjust vertical position: if the bottom overflows, place above the cursor.
        if (boxRect.bottom > window.innerHeight) {
            infoBox.style.top = (mousePosition.y - boxRect.height - offset) + 'px';
        }
        // Similarly, if the top is off screen, position below the cursor.
        if (boxRect.top < 0) {
            infoBox.style.top = (mousePosition.y + offset) + 'px';
        }
    }

    // Updated hideCompressedInfo to clear and hide the info box.
    function hideCompressedInfo() {
        infoBox.style.display = 'none';
        infoBox.innerHTML = '';
        infoBox.classList.remove('infoBox--hover');
    }

    // // Re-enable left-click so that when a satellite is clicked, its orbit is toggled.
    // viewer.screenSpaceEventHandler.setInputAction(function onLeftClick(movement) {
    //     const pickedObject = viewer.scene.pick(movement.position);
    //     if (Cesium.defined(pickedObject) && Cesium.defined(pickedObject.id)) {
    //         toggleOrbit(pickedObject.id);
    //     } else {
    //         removeAllEntityPaths();
    //     }
    // }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    viewer.screenSpaceEventHandler.setInputAction(function onMouseMove(movement) {
        const pickedObject = viewer.scene.pick(movement.endPosition);
        if (Cesium.defined(pickedObject) && pickedObject.id) {
            showCompressedInfo(pickedObject.id, movement.endPosition);
        } else {
            hideCompressedInfo();
        }
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

    // viewer.screenSpaceEventHandler.setInputAction(function onLeftClick(movement) {
    //     const pickedObject = viewer.scene.pick(movement.position);
    //     if (Cesium.defined(pickedObject) && Cesium.defined(pickedObject.id)) {
    //         const entity = pickedObject.id;
    //         showEntityPath(entity);
    //         highlightedEntities.push(entity);
    //     } else {
    //         infoBox.style.display = 'none';
    //         // Do nothing when clicking on the environment
    //     }
    // }, Cesium.ScreenSpaceEventType.LEFT_CLICK);


    // ensure all entities are not shown

    // initialise the model in cluster mode (default)
    removeEntities();
    document.getElementById('radio-leo').checked = true;
    void (async () => {
        enterClusteringPlaceholderView();
        // Keep the loading screen up briefly after data is ready.
        await delay(3000);
        document.body.classList.remove('orbx-app-loading');
        if (modelModePanel) {
            modelModePanel.classList.add('is-ready');
        }
        if (loadingScreen) {
            loadingScreen.style.display = 'none';
        }
    })();

    function isOrbitVisible(entity) {
        return !!(entity && entity.orbxOrbitVisible);
    }

    function syncOrbitRowVisibility(entityId) {
        const entity =
            dataSource && dataSource.entities && dataSource.entities.getById
                ? dataSource.entities.getById(entityId)
                : null;
        const isVisible = isOrbitVisible(entity);
        document.querySelectorAll(`tr[data-id="${entityId}"]`).forEach((row) => {
            row.classList.toggle('orbit-row-hidden', !isVisible);
        });
    }

    // Define toggleOrbit to show/hide the orbit path.
    function toggleOrbit(entityId, color) {
        const entity = dataSource && dataSource.entities && dataSource.entities.getById 
            ? dataSource.entities.getById(entityId)
            : null;
        if (!entity) return;
        if (isOrbitVisible(entity)) {
            removeEntityPath(entity);
        } else {
            showEntityPath(entity, color);
        }
        syncOrbitRowVisibility(entityId);
    }
    window.toggleOrbit = toggleOrbit;

    function getOrbitRingEntity(entity) {
        if (!entity || !dataSource || !dataSource.entities) return null;
        const id = String(entity.id || '');
        if (!id || id.endsWith('-orbit-ring')) return null;
        // Synthetics carry the orbit polyline on the SYN_* entity itself.
        if (id.startsWith('SYN_') && entity.polyline) return entity;
        return dataSource.entities.getById(`${id}-orbit-ring`) || null;
    }

    /** Bare NORAD / SYN_* entity that holds uniqueness / neighbours metadata. */
    function getPropertySourceEntity(entity, time) {
        if (!entity) return null;
        const id = String(entity.id || '');
        if (id.endsWith('-orbit-ring')) {
            const parentId =
                readEntityProperty(entity, 'parent_norad', time) ||
                id.slice(0, -'-orbit-ring'.length);
            const parent =
                dataSource && dataSource.entities
                    ? dataSource.entities.getById(String(parentId))
                    : null;
            return parent || entity;
        }
        return entity;
    }

    function readEntityProperty(entity, name, time) {
        if (!entity || !entity.properties) return undefined;
        const t = time || viewer.clock.currentTime;
        let prop = entity.properties[name];
        if (prop === undefined || prop === null) {
            // Some Cesium builds expose custom props only via getValue bag.
            if (typeof entity.properties.getValue === 'function') {
                const bag = entity.properties.getValue(t);
                if (bag && Object.prototype.hasOwnProperty.call(bag, name)) {
                    return bag[name];
                }
            }
            return undefined;
        }
        if (typeof prop.getValue === 'function') {
            return prop.getValue(t);
        }
        return prop;
    }

    function readNumericEntityProperty(entity, name, time) {
        const source = getPropertySourceEntity(entity, time);
        const raw = readEntityProperty(source, name, time);
        if (raw === null || raw === undefined) return null;
        if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
        if (typeof raw === 'string') {
            const trimmed = raw.trim();
            if (!trimmed || trimmed.toLowerCase() === 'none') return null;
            const n = Number(trimmed);
            return Number.isFinite(n) ? n : null;
        }
        const n = Number(raw);
        return Number.isFinite(n) ? n : null;
    }

    function formatUniquenessScore(entity, time) {
        const uniqueness = readNumericEntityProperty(entity, 'uniqueness', time);
        if (uniqueness === null) return 'N/A';
        if (uniqueness < 0.01) return uniqueness.toExponential(2);
        return uniqueness.toFixed(2);
    }

    function setOrbitRingVisible(entity, visible, color) {
        const ring = getOrbitRingEntity(entity);
        if (!ring) return;
        ring.show = !!visible;
        if (ring.polyline) {
            ring.polyline.show = !!visible;
            if (visible && color) {
                ring.polyline.material = new Cesium.ColorMaterialProperty(color);
                ring.polyline.width = 2;
            }
        }   
    }

    function showEntityPath(entity, color=undefined) {
        // Use the passed color, or the saved color, otherwise default to white.
        const orbit_color = color || entity.orbitColor || Cesium.Color.WHITE;
        // Store the color on the entity for later toggling.
        entity.orbitColor = orbit_color;

        // Orbits only: strip live trails and point markers.
        entity.path = undefined;
        entity.point = undefined;
        if (entity.billboard) {
            entity.billboard = undefined;
        }
        if (entity.label) {
            entity.label = undefined;
        }

        // Frozen common-epoch orbit ring = shape comparison geometry.
        setOrbitRingVisible(entity, true, orbit_color);
        entity.orbxOrbitVisible = true;
        entity.show = true;
        if (!visibleOrbitEntities.includes(entity)) {
            visibleOrbitEntities.push(entity);
        }
    }

    function removeEntityPath(entity) {
        if (!entity) return;
        if (entity.path) {
            entity.path = undefined;
        }
        entity.orbxOrbitVisible = false;
        setOrbitRingVisible(entity, false);
        entity.show = false;
        visibleOrbitEntities = visibleOrbitEntities.filter((e) => e !== entity);
    }

    function removeAllEntityPaths() {
        const pending = visibleOrbitEntities.slice();
        visibleOrbitEntities = [];
        pending.forEach((entity) => {
            if (!entity) return;
            if (entity.path) {
                entity.path = undefined;
            }
            entity.orbxOrbitVisible = false;
            setOrbitRingVisible(entity, false);
            entity.show = false;
        });
        highlightedEntities = [];
    }

    /** Hard-hide every orbit polyline, not just the ones we think are visible. */
    function hideAllOrbitGeometry() {
        visibleOrbitEntities = [];
        highlightedEntities = [];
        if (!dataSource || !dataSource.entities) {
            return;
        }
        dataSource.entities.values.forEach((entity) => {
            if (!entity) return;
            entity.path = undefined;
            entity.orbxOrbitVisible = false;
            entity.show = false;
            if (entity.polyline) {
                entity.polyline.show = false;
            }
        });
    }

    function removeEntities() {
        hideAllOrbitGeometry();
    }

    function hideUiPanel(panel) {
        if (!panel) return;
        panel.classList.remove('is-styled-ready');
        panel.style.display = 'none';
    }

    /** Show a panel after layout so CSS zoom does not flash unscaled content. */
    function showUiPanel(panel, html) {
        if (!panel) return;
        panel.classList.remove('is-styled-ready');
        if (html !== undefined) {
            panel.innerHTML = html;
        }
        panel.style.display = 'block';
        void panel.offsetWidth;
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                if (panel.style.display !== 'none') {
                    panel.classList.add('is-styled-ready');
                }
            });
        });
    }

    function getEntityAvailabilityInterval(entity) {
        if (!entity || !entity.availability || entity.availability.length === 0) {
            return null;
        }
        const interval = entity.availability.get(0);
        if (!interval || !interval.start) return null;
        return interval;
    }

    /**
     * Jump viewer clock into a shared cluster availability window so PathGraphics
     * resolve (common-time CZML: all members share one display epoch).
     *
     * PathGraphics only draws lead/trail around currentTime. Parking near
     * availability start leaves almost no past samples → short strips.
     * Place the clock far enough in that trailTime (~5600 s) has history.
     */
    function syncClockToEntities(entities, preferEntity) {
        const list = (entities || []).filter(Boolean);
        if (preferEntity) {
            list.unshift(preferEntity);
        }
        list.sort((a, b) => {
            const rank = (e) => {
                const id = String(e.id || '');
                if (id.startsWith('SYN_frechet_')) return 0;
                if (id.startsWith('SYN_max_separation_')) return 1;
                return 2;
            };
            return rank(a) - rank(b);
        });

        for (const entity of list) {
            const interval = getEntityAvailabilityInterval(entity);
            if (!interval) continue;

            const start = interval.start;
            const stop = interval.stop;
            const span = Math.max(
                60,
                Cesium.JulianDate.secondsDifference(stop, start)
            );

            // Need enough history for the short PathGraphics trail (~900 s).
            const trailSeconds = 900;
            const desiredOffset = Math.min(
                Math.max(trailSeconds + 60, span * 0.5),
                Math.max(60, span - 60)
            );

            viewer.clock.currentTime = Cesium.JulianDate.addSeconds(
                start,
                desiredOffset,
                new Cesium.JulianDate()
            );
            viewer.clock.startTime = start.clone();
            viewer.clock.stopTime = stop.clone();
            viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
            viewer.clock.multiplier = 1;

            if (viewer.timeline) {
                viewer.timeline.zoomTo(viewer.clock.startTime, viewer.clock.stopTime);
            }
            return true;
        }
        return false;
    }

    function syncClockToDataAvailability() {
        if (!dataSource) return;

        // Prefer entity availability (common display epoch). Ion packages often
        // pin dataSource.clock to "now", outside the CZML window.
        if (syncClockToEntities(dataSource.entities.values)) {
            return;
        }

        if (dataSource.clock) {
            dataSource.clock.getValue(viewer.clock);
        }
    }

    function getEntityPositions(entities, time) {
        const positions = [];
        (entities || []).forEach((entity) => {
            if (!entity) return;
            let got = false;
            if (entity.position && typeof entity.position.getValue === 'function') {
                try {
                    const position = entity.position.getValue(time);
                    if (Cesium.defined(position)) {
                        positions.push(position);
                        got = true;
                    }
                } catch (_) {
                    /* ignore */
                }
            }
            // Fallback: frozen orbit ring vertices (clock-independent).
            if (!got) {
                const ring = getOrbitRingEntity(entity);
                const poly = (ring && ring.polyline) || entity.polyline;
                if (poly && poly.positions) {
                    try {
                        const polyPos = poly.positions.getValue(time);
                        if (Cesium.defined(polyPos) && polyPos.length) {
                            for (let i = 0; i < polyPos.length; i++) {
                                if (Cesium.defined(polyPos[i])) {
                                    positions.push(polyPos[i]);
                                }
                            }
                        }
                    } catch (_) {
                        /* ignore */
                    }
                }
            }
        });
        return positions;
    }

    /**
     * Always-on camera move for newly shown satellites.
     * viewer.flyTo often no-ops when path bounding spheres aren't ready;
     * flying to an explicit sphere from current positions is reliable.
     */
    async function flyToEntities(targets, options = {}) {
        const entities = (Array.isArray(targets) ? targets : [targets]).filter(Boolean);
        if (entities.length === 0) return;

        viewer.resize();

        syncClockToEntities(entities, options.preferEntity);

        let time = viewer.clock.currentTime;
        let positions = getEntityPositions(entities, time);
        if (positions.length === 0) {
            // One frame later positions sometimes become available after show=true.
            await new Promise((resolve) => requestAnimationFrame(resolve));
            time = viewer.clock.currentTime;
            positions = getEntityPositions(entities, time);
        }
        if (positions.length === 0) {
            console.warn('[OrbX] flyTo skipped — no entity positions at clock time');
            return;
        }

        const bs = Cesium.BoundingSphere.fromPoints(positions);
        // Position-only spheres are often tight for LEO / small clusters.
        // Keep a higher floor so the framing stays more Earth-wide.
        const range = Math.max(bs.radius * 4.5, 2.2e7);
        const duration = options.duration ?? 1.5;

        await new Promise((resolve) => {
            viewer.camera.flyToBoundingSphere(bs, {
                duration,
                offset: new Cesium.HeadingPitchRange(
                    0,
                    Cesium.Math.toRadians(-90),
                    range
                ),
                complete: resolve,
                cancel: resolve,
            });
        });
    }

    /** Fréchet / max-separation rows in CZML (SYN_* ids or synthetic_type property). */
    function isSyntheticEntity(entity, time) {
        if (!entity) return false;
        const id = String(entity.id || '');
        if (id.startsWith('SYN_')) return true;

        if (!entity.properties) return false;
        let raw = entity.properties.synthetic_type;
        if (raw === undefined || raw === null) return false;
        if (typeof raw.getValue === 'function') {
            raw = raw.getValue(time);
        }
        if (raw === null || raw === undefined) return false;
        const normalized = String(raw).trim().toLowerCase();
        return normalized === 'frechet' || normalized === 'max_separation';
    }

    /** Hover / table NORAD label: synthetics → 99999; strip -orbit-ring from reals. */
    function formatHoverNoradId(entity, time) {
        if (!entity) return 'N/A';
        if (isSyntheticEntity(entity, time)) return '99999';
        let id = String(entity.id || '');
        if (id.endsWith('-orbit-ring')) {
            id = id.slice(0, -'-orbit-ring'.length);
        }
        return id || 'N/A';
    }

    function formatDisplayNoradFromId(entityId) {
        let id = String(entityId || '');
        if (id.startsWith('SYN_')) return '99999';
        if (id.endsWith('-orbit-ring')) {
            id = id.slice(0, -'-orbit-ring'.length);
        }
        return id || 'N/A';
    }

    function getBareEntityId(entityId) {
        let id = String(entityId || '');
        if (id.endsWith('-orbit-ring')) {
            id = id.slice(0, -'-orbit-ring'.length);
        }
        return id;
    }

    function isRealSatelliteEntity(entity, time) {
        return !!entity && !isSyntheticEntity(entity, time);
    }

    function isLookupRealSatelliteEntity(entity, time) {
        if (!isRealSatelliteEntity(entity, time)) return false;
        const id = String(entity.id || '');
        return !!id && !id.endsWith('-orbit-ring');
    }

    function getSyntheticTypeFromEntity(entity, time) {
        if (!entity || !entity.properties) return null;
        let raw = entity.properties.synthetic_type;
        if (raw === undefined || raw === null) return null;
        if (typeof raw.getValue === 'function') {
            raw = raw.getValue(time);
        }
        if (raw === null || raw === undefined) return null;
        const normalized = String(raw).trim().toLowerCase();
        if (normalized === 'none' || normalized === '') return null;
        if (normalized === 'frechet' || normalized === 'max_separation') {
            return normalized;
        }
        return null;
    }

    function getClusterRoleLabel(entity, time) {
        const synthType = getSyntheticTypeFromEntity(entity, time);
        if (synthType === 'frechet') return 'Fréchet Mean';
        if (synthType === 'max_separation') return 'Max-separation';
        return 'Member';
    }

    /** Cluster mode: both Fréchet and max-separation synthetics exist for this label. */
    function clusterHasSyntheticPair(clusterLabel) {
        if (clusterLabel === null || clusterLabel === -1) return false;
        const lab = Number(clusterLabel);
        if (!Number.isFinite(lab)) return false;

        if (dataSource && dataSource.entities && typeof dataSource.entities.getById === 'function') {
            const hasFrechet = !!dataSource.entities.getById(`SYN_frechet_${lab}`);
            const hasMaxSep = !!dataSource.entities.getById(`SYN_max_separation_${lab}`);
            if (hasFrechet && hasMaxSep) return true;
        }

        const members = clusterLabelToEntities.get(lab) || [];
        const now = Cesium.JulianDate.now();
        let hasFrechet = false;
        let hasMaxSep = false;
        for (const entity of members) {
            const st = getSyntheticTypeFromEntity(entity, now);
            if (st === 'frechet') hasFrechet = true;
            if (st === 'max_separation') hasMaxSep = true;
            if (hasFrechet && hasMaxSep) return true;
        }
        return false;
    }

    function getClusterLabelFromEntity(entity, time) {
        if (!entity || !entity.properties) return null;
        let raw = entity.properties.cluster_label;
        if (raw === undefined || raw === null) return null;
        if (typeof raw.getValue === 'function') {
            raw = raw.getValue(time);
        }
        if (raw === null || raw === undefined) return null;
        const n = Number(raw);
        return Number.isFinite(n) ? n : null;
    }

    function getClusterDensityFromEntity(entity, time) {
        if (!entity || !entity.properties) return null;
        let raw = entity.properties.cluster_density;
        if (raw === undefined || raw === null) return null;
        if (typeof raw.getValue === 'function') {
            raw = raw.getValue(time);
        }
        if (raw === null || raw === undefined) return null;
        if (String(raw).trim().toLowerCase() === 'none') return null;
        const n = Number(raw);
        return Number.isFinite(n) ? n : null;
    }

    function formatClusterDensity(density) {
        if (density === null || density === undefined) return 'N/A';
        if (typeof density !== 'number') return String(density);
        if (density < 0.01) return density.toExponential(2);
        return density.toFixed(4);
    }

    function getClusterDensityFromMembers(members) {
        const now = Cesium.JulianDate.now();
        for (const entity of members) {
            const d = getClusterDensityFromEntity(entity, now);
            if (d !== null) return d;
        }
        return null;
    }

    function rebuildClusterIndex() {
        clusterLabelToEntities.clear();
        if (!dataSource || !dataSource.entities) return;
        const now = Cesium.JulianDate.now();
        dataSource.entities.values.forEach((entity) => {
            if (String(entity.id || '').endsWith('-orbit-ring')) return;
            const lab = getClusterLabelFromEntity(entity, now);
            if (lab === null) return;
            if (!clusterLabelToEntities.has(lab)) {
                clusterLabelToEntities.set(lab, []);
            }
            clusterLabelToEntities.get(lab).push(entity);
        });
    }

    /** Cluster size bands: micro 2–3, minor 4–8, major 9–15, mega 16+ (real satellites only). */
    function clusterSizeTier(memberCount) {
        if (memberCount >= 2 && memberCount <= 3) return 'micro';
        if (memberCount >= 4 && memberCount <= 8) return 'minor';
        if (memberCount >= 9 && memberCount <= 15) return 'major';
        if (memberCount >= 16) return 'mega';
        return null;
    }

    function getClusterRealMemberCount(entities, time) {
        if (!entities || entities.length === 0) return 0;
        const t = time !== undefined ? time : Cesium.JulianDate.now();
        return entities.filter((entity) => isRealSatelliteEntity(entity, t)).length;
    }

    function tierBandLabel(tier) {
        const map = {
            micro: 'Micro (2–3 satellites)',
            minor: 'Minor (4–8 satellites)',
            major: 'Major (9–15 satellites)',
            mega: 'Mega (16+ satellites)'
        };
        return map[tier] || tier || '';
    }

    function pickCandidateClusterLabel(candidates, excludedLabel = currentClusterLabel) {
        if (!candidates || candidates.length === 0) return null;
        const filtered = candidates.filter((label) => label !== excludedLabel);
        const pool = filtered.length > 0 ? filtered : candidates;
        return pool[Math.floor(Math.random() * pool.length)];
    }

    function pickRandomClusterLabelForTier(category) {
        const candidates = [];
        clusterLabelToEntities.forEach((entities, label) => {
            if (label === -1) return;
            if (!clusterHasSyntheticPair(label)) return;
            const n = getClusterRealMemberCount(entities);
            if (clusterSizeTier(n) === category) {
                candidates.push(label);
            }
        });
        return pickCandidateClusterLabel(candidates);
    }

    /** Random cluster with both synthetic orbits; excludes noise (-1). */
    function pickRandomClusterLabelAny() {
        const candidates = [];
        clusterLabelToEntities.forEach((entities, label) => {
            if (label === -1) return;
            if (!clusterHasSyntheticPair(label)) return;
            candidates.push(label);
        });
        return pickCandidateClusterLabel(candidates);
    }

    const SYNTHETIC_PATH_COLOR_FRECHET = Cesium.Color.RED;
    const SYNTHETIC_PATH_COLOR_MAX_SEPARATION = Cesium.Color.BLUE;

    function getPathColorForClusterEntity(entity, _highlightEntity) {
        const now = Cesium.JulianDate.now();
        const synthType = getSyntheticTypeFromEntity(entity, now);
        if (synthType === 'frechet') return SYNTHETIC_PATH_COLOR_FRECHET;
        if (synthType === 'max_separation') return SYNTHETIC_PATH_COLOR_MAX_SEPARATION;
        return Cesium.Color.fromCssColorString('#20c997');
    }

    async function displayClusterByLabel(clusterLabel, highlightEntity, options = {}) {
        const generation = ++clusterDisplayGeneration;
        if (!clusterHasSyntheticPair(clusterLabel)) {
            console.warn('[OrbX] Cluster has no synthetic orbit pair:', clusterLabel);
            return;
        }
        // Full sweep — tracked-list clears alone can miss leftovers under race.
        hideAllOrbitGeometry();
        viewer.scene.requestRender();
        // Give Cesium a beat to clear the previous rings before drawing the next set.
        await delay(100);
        if (generation !== clusterDisplayGeneration) {
            return;
        }
        const members = clusterLabelToEntities.get(clusterLabel);
        if (!members || members.length === 0) {
            console.warn('[OrbX] No entities for cluster label', clusterLabel);
            return;
        }
        // Hide again immediately before showing, in case another switch interleaved.
        hideAllOrbitGeometry();
        if (generation !== clusterDisplayGeneration) {
            return;
        }
        currentClusterLabel = clusterLabel;
        currentClusterHighlightId = highlightEntity
            ? String(highlightEntity.id)
            : null;
        // Keep size-band radios in sync when search/random lands on a cluster
        // (category clicks already match; re-checking is harmless).
        const tier = clusterSizeTier(getClusterRealMemberCount(members));
        if (tier) {
            setSelectedClusterCategory(tier);
        }
        const prefer =
            highlightEntity ||
            members.find((e) => String(e.id).startsWith('SYN_frechet_')) ||
            members[0];
        syncClockToEntities(members, prefer);
        members.forEach((entity) => {
            showEntityPath(entity, getPathColorForClusterEntity(entity, highlightEntity));
        });
        displayClusterMemberList(clusterLabel, members, highlightEntity);
        if (!options.skipFlyTo) {
            await flyToEntities(members, { duration: 2, preferEntity: prefer });
        }
        // If a newer cluster switch started during flyTo, it owns the scene now.
        if (generation !== clusterDisplayGeneration) {
            return;
        }
    }

    async function pickAndShowRandomClusterForCategory(category) {
        rebuildClusterIndex();
        const clusterId = pickRandomClusterLabelForTier(category);
        if (clusterId === null) {
            alert(
                'No clusters in this size band that include synthetic orbits (Fréchet + max-separation). Try another category.'
            );
            return;
        }
        await displayClusterByLabel(clusterId);
    }

    function getOrbitEntities(selectedOrbit){
        const entities = dataSource.entities.values;

        const orbitEntities = entities.filter((entity) => {
            if (!isLookupRealSatelliteEntity(entity)) return false;
            const orbit_class = readEntityProperty(entity, 'orbit_class');
            return orbit_class === selectedOrbit;
        });
        return orbitEntities;
    }

    function getSelectedOrbit(){
        let orbit = "";
        if (document.getElementById('radio-leo').checked) orbit = "LEO";
        if (document.getElementById('radio-meo').checked) orbit = "MEO";
        if (document.getElementById('radio-geo').checked) orbit = "GEO";
        if (document.getElementById('radio-heo').checked) orbit = "HEO";
        console.log("getSelectedOrbit called: ", orbit);
        return orbit;
    }

    function getTopBottomEntities(entities){
        // check that entities is an array:
        // if (!Array.isArray(entities) && entities.length === 0) {
        //     throw new Error('entities must be an array or is empty');
        // } else {
        //     console.log("getTopBottomEntities called, entities is a valid array");
        // }

        // console.log("number of entities: ", entities.length);
        // sort the entities and get the top and bottom 5
        entities.sort((a, b) => {
            const ra = readNumericEntityProperty(a, 'rank');
            const rb = readNumericEntityProperty(b, 'rank');
            return (ra ?? Number.POSITIVE_INFINITY) - (rb ?? Number.POSITIVE_INFINITY);
        });

        const topEntities = entities.slice(0, 5);
        const bottomEntities = entities.slice(-5);

        // Show in ascending order
        bottomEntities.reverse();

        if (topEntities.length !== 5 || bottomEntities.length !== 5) {
            throw new Error('topEntities and bottomEntities must have 5 entities each');
        }

        return [topEntities, bottomEntities];
    }

    // will return the top and bottom 5 entities based on uniqueness rank for the given orbit
    async function showUniqueOrbits() {
        // get which orbit radio is selected
        const selectedOrbit = getSelectedOrbit();
        // get the entities in the selected orbit
        const entities = getOrbitEntities(selectedOrbit);
    
        const [topEntities, bottomEntities] = getTopBottomEntities(entities);
        
        // remove all entity paths
        removeAllEntityPaths();
    
        if(topEntities.length === 0 && bottomEntities.length === 0) {
            throw new Error('topEntities and bottomEntities must have 5 entities each');
        }
    
        topEntities.forEach(entity => showEntityPath(entity, Cesium.Color.RED));
        bottomEntities.forEach(entity => showEntityPath(entity, Cesium.Color.GREEN));

        uniqueModeShowingNeighbours = false;
        updateRankingsDisplay(topEntities, bottomEntities);

        // Zoom in on the displayed satellites
        await flyToEntities([...topEntities, ...bottomEntities], { duration: 1 });
    }

    // Changing regime always restores the default top/bottom view.
    // Re-activating the *same* regime (radio or label text) does so only
    // after a neighbour search.
    ['radio-leo', 'radio-meo', 'radio-geo', 'radio-heo'].forEach(id => {
        const radio = document.getElementById(id);
        if (!radio) return;
        const option = radio.closest('.orbit-option') || radio;

        let wasCheckedBeforeActivate = false;
        option.addEventListener('pointerdown', function() {
            wasCheckedBeforeActivate = radio.checked;
        });
        option.addEventListener('click', function() {
            if (!document.getElementById('radio-mode-unique').checked) {
                return;
            }
            if (!wasCheckedBeforeActivate || !uniqueModeShowingNeighbours) {
                return;
            }
            console.log("orbit regime re-select restore:", id);
            void handleOrbitToggle();
        });
        radio.addEventListener('change', function() {
            if (!document.getElementById('radio-mode-unique').checked) {
                return;
            }
            console.log("orbit regime change:", id);
            void handleOrbitToggle();
        });
    });

    function clearClusterCategoryRadios() {
        ['radio-micro', 'radio-minor', 'radio-major', 'radio-mega'].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.checked = false;
        });
    }

    function getSelectedClusterCategory() {
        if (document.getElementById('radio-micro').checked) return 'micro';
        if (document.getElementById('radio-minor').checked) return 'minor';
        if (document.getElementById('radio-major').checked) return 'major';
        if (document.getElementById('radio-mega').checked) return 'mega';
        return null;
    }

    function clusterCategoryFromRadioId(radioId) {
        const map = {
            'radio-micro': 'micro',
            'radio-minor': 'minor',
            'radio-major': 'major',
            'radio-mega': 'mega'
        };
        return map[radioId] || 'micro';
    }

    // Use click (not change) so clicking the same category again samples another cluster.
    ['radio-micro', 'radio-minor', 'radio-major', 'radio-mega'].forEach(id => {
        const radio = document.getElementById(id);
        if (radio) {
            radio.addEventListener('click', function() {
                if (!document.getElementById('radio-mode-clusters').checked) {
                    return;
                }
                const category = clusterCategoryFromRadioId(id);
                console.log('[OrbX] Cluster category:', category);
                void pickAndShowRandomClusterForCategory(category);
            });
        }
    });


    async function performUniqueModeSearch(searchId) {
        if (!searchId) {
            console.log("No search ID provided");
            return;
        }
        try {
            if (searchId.toLowerCase() === 'random') {
                const realEntities = dataSource.entities.values.filter((e) =>
                    isLookupRealSatelliteEntity(e)
                );
                if (realEntities.length === 0) {
                    alert('No real satellites available in the data source.');
                    return;
                }
                const randomIndex = Math.floor(Math.random() * realEntities.length);
                searchId = realEntities[randomIndex].id;
            }

            searchId = getBareEntityId(searchId);
            const searchedEntity = getEntityFromId(searchId);
            if (!searchedEntity) {
                alert("NORAD ID not found in data source");
                return;
            }
            if (isSyntheticEntity(searchedEntity)) {
                alert(
                    'That ID is a synthetic orbit (Fréchet / max-separation). ' +
                    'Use Orbital clusters mode, or search a real NORAD ID.'
                );
                return;
            }

            const orbitClass = readEntityProperty(searchedEntity, 'orbit_class');
            if (orbitClass) {
                setSelectedOrbit(orbitClass);
            }

            const neighbourIds = readEntityProperty(searchedEntity, 'neighbours');
            console.log("neighbourIds: ", neighbourIds);

            const neighbourEntities = [];
            if (neighbourIds) {
                const neighbourIdArray = Object.values(neighbourIds);
                neighbourIdArray.forEach(neighbourId => {
                    const neighbourEntity = getEntityFromId(neighbourId);
                    if (neighbourEntity && isLookupRealSatelliteEntity(neighbourEntity)) {
                        neighbourEntities.push(neighbourEntity);
                    }
                });
            }

            const searchResults = document.getElementById('searchResults');
            hideUiPanel(topBottomInfoBox);
            hideClusterMemberList();

            if (!neighbourEntities || neighbourEntities.length === 0) {
                console.log("No neighbours found for NORAD ID: " + formatDisplayNoradFromId(searchId));
                if (searchResults) {
                    showUiPanel(
                        searchResults,
                        `<p>No neighbours found for NORAD ID: ${formatDisplayNoradFromId(searchId)}</p>`
                    );
                }
                return;
            }

            if (searchResults) {
                console.log("searchResults found");
                showUiPanel(
                    searchResults,
                    generateNeighbourSatelliteList({
                        targetId: searchId,
                        list: neighbourEntities,
                    })
                );
                attachNeighbourLinkHandlers('.neighbour-list-container .satellite-id');
                attachOrbitToggleRowHandlers('.neighbour-row');
            }

            removeAllEntityPaths();
            removeEntities();

            neighbourEntities.forEach(neighbour => showEntityPath(neighbour, Cesium.Color.YELLOW));
            showEntityPath(searchedEntity, Cesium.Color.BLUE);
            uniqueModeShowingNeighbours = true;

            await flyToEntities([...neighbourEntities, searchedEntity], { duration: 2 });

            console.log("You should see results now");
        } catch (error) {
            console.error(error);
        }
    }

    async function performClusterModeSearch(searchId) {
        if (!searchId || !dataSource || !dataSource.entities) {
            console.log("No search ID or data source");
            return;
        }
        try {
            rebuildClusterIndex();

            if (searchId.toLowerCase() === 'random') {
                const clusterId = pickRandomClusterLabelAny();
                if (clusterId === null) {
                    alert(
                        'No clusters with synthetic orbits in the dataset (Fréchet + max-separation pairs).'
                    );
                    return;
                }
                await displayClusterByLabel(clusterId);
                return;
            }

            const searchedEntity = dataSource.entities.getById(searchId);
            const searchResults = document.getElementById('searchResults');
            hideUiPanel(clusterMemberListBox);

            const showNotInClusterModel = () => {
                hideClusterMemberList();
                hideUiPanel(topBottomInfoBox);
                removeAllEntityPaths();
                removeEntities();
                if (searchResults) {
                    showUiPanel(
                        searchResults,
                        `<div style="padding:12px;background:rgba(30,30,30,0.85);color:#eee;border-radius:8px;max-width:420px;font-family:Arial,sans-serif;font-size:14px;">` +
                        `<strong>NORAD ${searchId}</strong> does not exist in the cluster model.<br>` +
                        `This mode only includes satellites that exist in the 300-700km altitude range. ` +
                        `Try the Unique orbits mode, or search a clustered NORAD ID.` +
                        `</div>`
                    );
                }
            };

            if (!searchedEntity) {
                showNotInClusterModel();
                return;
            }

            const now = Cesium.JulianDate.now();
            const lab = getClusterLabelFromEntity(searchedEntity, now);

            if (lab === null || lab === -1) {
                showNotInClusterModel();
                return;
            }

            removeAllEntityPaths();
            removeEntities();

            if (!clusterHasSyntheticPair(lab)) {
                if (searchResults) {
                    showUiPanel(
                        searchResults,
                        `<div style="padding:12px;background:rgba(30,30,30,0.85);color:#eee;border-radius:8px;max-width:420px;font-family:Arial,sans-serif;font-size:14px;">` +
                        `<strong>NORAD ${searchId}</strong><br>` +
                        `Cluster ID: <strong>${lab}</strong><br>` +
                        `This cluster has no synthetic orbits in the dataset (Fréchet + max-separation not generated for this cluster).` +
                        `</div>`
                    );
                }
                showEntityPath(
                    searchedEntity,
                    Cesium.Color.fromCssColorString('#20c997')
                );
                await flyToEntities(searchedEntity, { duration: 2 });
                return;
            }

            await displayClusterByLabel(lab, searchedEntity);
        } catch (error) {
            console.error(error);
        }
    }

    async function performSearch(searchId) {
        if (!searchId) {
            console.log("No search ID provided");
            return;
        }
        if (document.getElementById('radio-mode-clusters').checked) {
            return performClusterModeSearch(searchId);
        }
        return performUniqueModeSearch(searchId);
    }

    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const randomBtn = document.getElementById('randomBtn');
    const fakePlaceholder = document.getElementById("fakePlaceholder");

    // Search button functionality
    searchBtn.addEventListener('click', () => {
        performSearch(searchInput.value.trim());
    });

    // Random button functionality
    randomBtn.addEventListener('click', () => {
        // Enable search input in case it was disabled
        searchInput.disabled = false;
        searchInput.style.backgroundColor = '';
        searchInput.value = '';
        
        // Perform random search
        performSearch('random');
    });

    // Allow pressing Enter in the search input
    searchInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            performSearch(searchInput.value.trim());
        }
    });

    // Placeholder animation
    const placeholders = [
        "62922 (STARLINK)",
        "25544 (ISS)",
        "20580 (HST)"
    ];

    const clusterPlaceholders = [
        "57070 (STARLINK)",
        "12230 (DELTA 1 DEB)",
        "47971 (GLOBAL-9)",
        "28494 (ESSAIM-1)",
        "24893 (SL-3 DEB)",
        "39769 (RISING 2)"
    ];

    let index = 0;

    setInterval(() => {
        if (document.activeElement !== searchInput && searchInput.value.trim() === "") {
            fakePlaceholder.classList.add("fade-out");
            setTimeout(() => {
                fakePlaceholder.textContent = placeholders[index];
                index = (index + 1) % placeholders.length;
                fakePlaceholder.classList.remove("fade-out");
            }, 500);
        }
    }, 4000);

    searchInput.addEventListener('focus', () => {
        fakePlaceholder.style.visibility = "hidden";
    });

    searchInput.addEventListener('blur', () => {
        if (searchInput.value.trim() === "") {
            fakePlaceholder.style.visibility = "visible";
        }
    });

    searchInput.addEventListener('input', () => {
        if (searchInput.value.trim() !== "") {
            fakePlaceholder.textContent = "";
        } else if (document.activeElement !== searchInput) {
            fakePlaceholder.style.visibility = "visible";
        }
    });

    const clusterSearchInput = document.getElementById('clusterSearchInput');
    const clusterSearchBtn = document.getElementById('clusterSearchBtn');
    const clusterRandomBtn = document.getElementById('clusterRandomBtn');
    const clusterFakePlaceholder = document.getElementById('clusterFakePlaceholder');

    if (clusterSearchBtn && clusterSearchInput) {
        clusterSearchBtn.addEventListener('click', () => {
            performSearch(clusterSearchInput.value.trim());
        });
    }

    if (clusterRandomBtn && clusterSearchInput) {
        clusterRandomBtn.addEventListener('click', () => {
            clusterSearchInput.disabled = false;
            clusterSearchInput.style.backgroundColor = '';
            clusterSearchInput.value = '';
            performSearch('random');
        });
    }

    if (clusterSearchInput) {
        clusterSearchInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                performSearch(clusterSearchInput.value.trim());
            }
        });
    }

    if (clusterFakePlaceholder && clusterSearchInput) {
        let clusterPlaceholderIndex = 0;
        setInterval(() => {
            if (document.activeElement !== clusterSearchInput && clusterSearchInput.value.trim() === "") {
                clusterFakePlaceholder.classList.add("fade-out");
                setTimeout(() => {
                    clusterFakePlaceholder.textContent = clusterPlaceholders[clusterPlaceholderIndex];
                    clusterPlaceholderIndex = (clusterPlaceholderIndex + 1) % clusterPlaceholders.length;
                    clusterFakePlaceholder.classList.remove("fade-out");
                }, 500);
            }
        }, 4000);

        clusterSearchInput.addEventListener('focus', () => {
            clusterFakePlaceholder.style.visibility = "hidden";
        });

        clusterSearchInput.addEventListener('blur', () => {
            if (clusterSearchInput.value.trim() === "") {
                clusterFakePlaceholder.style.visibility = "visible";
            }
        });

        clusterSearchInput.addEventListener('input', () => {
            if (clusterSearchInput.value.trim() !== "") {
                clusterFakePlaceholder.textContent = "";
            } else if (document.activeElement !== clusterSearchInput) {
                clusterFakePlaceholder.style.visibility = "visible";
            }
        });
    }

    if (viewer.homeButton) {
        viewer.homeButton.viewModel.command.afterExecute.addEventListener(function() {
            removeAllEntityPaths();
            infoBox.style.display = 'none';
        });
    }

    function getEntityFromId(entityId){
        const lookupId = getBareEntityId(entityId);
        const entity = dataSource && dataSource.entities && typeof dataSource.entities.getById === 'function'
            ? dataSource.entities.getById(lookupId)
            : null;

        return entity;
    }

    function hideClusterMemberList() {
        hideUiPanel(clusterMemberListBox);
        if (clusterMemberListBox) {
            clusterMemberListBox.innerHTML = '';
        }
    }

    function displayClusterMemberList(clusterLabel, members, highlightEntity) {
        if (!clusterMemberListBox || !members || members.length === 0) return;

        hideUiPanel(document.getElementById('searchResults'));

        showUiPanel(
            clusterMemberListBox,
            renderClusterMemberList(clusterLabel, members, highlightEntity)
        );
        attachNeighbourLinkHandlers('.cluster-member-table .satellite-id');
        attachOrbitToggleRowHandlers('.cluster-member-row');
    }

    function sortClusterMembersForList(members) {
        const now = Cesium.JulianDate.now();
        const rank = (entity) => {
            const synthType = getSyntheticTypeFromEntity(entity, now);
            if (synthType === 'frechet') return 0;
            if (synthType === 'max_separation') return 1;
            return 2;
        };
        return [...members].sort((a, b) => {
            const ra = rank(a);
            const rb = rank(b);
            if (ra !== rb) return ra - rb;
            return String(a.id).localeCompare(String(b.id), undefined, { numeric: true });
        });
    }

    function generateClusterMemberRow(entity, index, highlightEntity) {
        const now = Cesium.JulianDate.now();
        const synthType = getSyntheticTypeFromEntity(entity, now);
        let roleLabel = getClusterRoleLabel(entity, now);
        let roleClass = 'cluster-role-member';
        if (synthType === 'frechet') {
            roleClass = 'cluster-role-frechet';
        } else if (synthType === 'max_separation') {
            roleClass = 'cluster-role-maxsep';
        }
        const isHighlight =
            highlightEntity && String(entity.id) === String(highlightEntity.id);
        const rowClass = isHighlight
            ? 'cluster-member-row cluster-member-row-highlight neighbour-row'
            : 'cluster-member-row neighbour-row';
        // Keep the real CZML entity id in data-id for path toggles; show 99999 for synthetics.
        const displayNorad = formatHoverNoradId(entity, now);

        return `
            <tr class="${rowClass}" data-id="${entity.id}">
                <td>${index + 1}</td>
                <td class="${roleClass}">${roleLabel}</td>
                <td><a href="#" class="satellite-id" data-id="${entity.id}">${displayNorad}</a></td>
                <td class="sat-name">${entity.name || 'N/A'}</td>
            </tr>
        `;
    }

    function renderClusterMemberList(clusterLabel, members, highlightEntity) {
        const sorted = sortClusterMembersForList(members);
        const realN = getClusterRealMemberCount(members);
        const tier = clusterSizeTier(realN);
        const tierText = tier
            ? tierBandLabel(tier)
            : `${realN} real satellites`;
        const clusterDensity = getClusterDensityFromMembers(members);
        const densityText = formatClusterDensity(clusterDensity);
        const rows = sorted
            .map((entity, index) => generateClusterMemberRow(entity, index, highlightEntity))
            .join('');

        return `
            <div class="container">
                <div class="rankings-card cluster-member-card">
                    <div class="card-header">
                        <h2 class="card-title">Cluster ${clusterLabel} · ${tierText} · Density ${densityText}</h2>
                    </div>
                    <table class="rankings-table cluster-member-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Role</th>
                                <th>NORAD ID</th>
                                <th>Satellite Name</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                    <div class="table-footer">
                        <div class="cluster-member-legend">
                            <span class="cluster-legend-item">
                                <span class="header-indicator red-indicator"></span>
                                Fréchet Mean Synthetic
                            </span>
                            <span class="cluster-legend-item">
                                <span class="header-indicator blue-indicator"></span>
                                Max-separation Synthetic
                            </span>
                            <span class="cluster-legend-item">
                                <span class="header-indicator teal-indicator"></span>
                                Cluster Member
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function generateSatelliteList(satellites) {
        return `<ul style="padding-left: 20px; list-style-type: none;">
            ${satellites.map(satellite => {
                const uniquenessStr = formatUniquenessScore(satellite);
                return `<li>
                    Score: <b>${uniquenessStr}</b> 
                    (<a href="#" class="satellite-id" data-id="${satellite.id}" style="cursor: pointer; color: blue; text-decoration: underline;">
                        ${satellite.id}
                    </a>)
                    ${satellite.name}
                </li>`;
            }).join('')}
        </ul>`;
    }

    function attachNeighbourLinkHandlers(selector = '.satellite-id') {
        console.log("called link handler")
        const links = document.querySelectorAll(selector);
        links.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const searchId = link.getAttribute('data-id');
                if (searchId) {
                    console.log("Performing search on:", searchId);
                    performSearch(searchId);
                } else {
                    console.error("No data-id found on the clicked element.");
                }
            });
        });
    }

    function attachOrbitToggleRowHandlers(selector = '.neighbour-row') {
        const rows = document.querySelectorAll(selector);
        rows.forEach(row => {
            row.addEventListener('click', (e) => {
                // Prevent the inner link click from also firing if needed:
                if (e.target.tagName.toLowerCase() !== 'a') {
                    e.preventDefault();
                    const noradId = row.getAttribute('data-id');
                    if (noradId) {
                        console.log("Toggling orbit for:", noradId);
                        toggleOrbit(noradId);
                    } else {
                        console.error("No data-id found on the row.");
                    }
                }
            });
        });
    }

    // In main.js-1, update displayTopAndBottomSatellitesByUniqueness:
    async function displayUniqueOrbitList() {
        console.log("displayUniqueOrbitList called");
        hideClusterMemberList();
        hideUiPanel(document.getElementById('searchResults'));
        
        const selectedOrbit = getSelectedOrbit();
        // get the top and bottom 5 entities
        const entities = getOrbitEntities(selectedOrbit);
        const [topEntities, bottomEntities] = getTopBottomEntities(entities);
    
        // Build the info box content using generateSatelliteList.
        let infoboxContent = `<h3><span class="box red"></span>5 Most Unique Orbits (${selectedOrbit})</h3>` + generateSatelliteList(topEntities);
        infoboxContent += `<h3><span class="box green"></span>5 Least Unique Orbits (${selectedOrbit})</h3>` + generateSatelliteList(bottomEntities);
    
        showUiPanel(topBottomInfoBox, infoboxContent);
        attachNeighbourLinkHandlers('.satellite-id');
    }

    function generateRankingRow(satellite, index) {
        const uniquenessStr = formatUniquenessScore(satellite);
        const displayNorad = formatDisplayNoradFromId(satellite.id);
        return `
            <tr>
                <td>${index + 1}</td>
                <td class="score-cell">${uniquenessStr}</td>
                <td><a href="#" class="satellite-id" data-id="${satellite.id}">${displayNorad}</a></td>
                <td class="sat-name">${satellite.name || "N/A"}</td>
            </tr>
        `;
    }
    
    function generateNeighbourRow(satellite, index) {
        const displayNorad = formatDisplayNoradFromId(satellite.id);
        return `
            <tr class="neighbour-row" data-id="${satellite.id}">
                <td>${index + 1}</td>
                <td><a href="#" class="satellite-id" data-id="${satellite.id}">${displayNorad}</a></td>
                <td class="neighbour-list-sat-name">${satellite.name}</td>
            </tr>
        `;
    }

    function setSelectedOrbit(orbit) {
        ['radio-leo', 'radio-meo', 'radio-geo', 'radio-heo'].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.checked = false;
        });
        const map = {
            LEO: 'radio-leo',
            MEO: 'radio-meo',
            GEO: 'radio-geo',
            HEO: 'radio-heo',
        };
        const target = document.getElementById(map[orbit]);
        if (target) target.checked = true;
    }

    function setSelectedClusterCategory(category) {
        clearClusterCategoryRadios();
        if (!category) return;
        const target = document.getElementById(`radio-${category}`);
        if (target) target.checked = true;
    }

    function serializeColor(color) {
        if (!color) return null;
        return {
            red: color.red,
            green: color.green,
            blue: color.blue,
            alpha: color.alpha,
        };
    }

    function deserializeColor(colorData) {
        if (!colorData) return undefined;
        return new Cesium.Color(
            colorData.red,
            colorData.green,
            colorData.blue,
            colorData.alpha
        );
    }

    function snapshotPathEntities() {
        if (!dataSource || !dataSource.entities) return [];
        return dataSource.entities.values
            .filter((entity) => isOrbitVisible(entity))
            .map((entity) => ({
                id: entity.id,
                color: serializeColor(entity.orbitColor),
            }));
    }

    function restorePathEntities(pathSnapshots) {
        removeAllEntityPaths();
        removeEntities();
        (pathSnapshots || []).forEach(({ id, color }) => {
            const entity = dataSource.entities.getById(id);
            if (entity) {
                showEntityPath(entity, deserializeColor(color));
            }
        });
    }

    function snapshotCamera() {
        return {
            position: viewer.camera.position.clone(),
            direction: viewer.camera.direction.clone(),
            up: viewer.camera.up.clone(),
        };
    }

    function restoreCamera(cameraState) {
        if (!cameraState) return;
        viewer.camera.position = cameraState.position.clone();
        viewer.camera.direction = cameraState.direction.clone();
        viewer.camera.up = cameraState.up.clone();
    }

    function restoreUiPanel(panel, html, display) {
        if (!panel) return;
        if (display === 'block') {
            showUiPanel(panel, html || '');
        } else {
            if (html !== undefined) {
                panel.innerHTML = html || '';
            }
            hideUiPanel(panel);
        }
    }

    function attachPanelLinkHandlers() {
        attachNeighbourLinkHandlers('.satellite-id');
        attachNeighbourLinkHandlers('.neighbour-list-container .satellite-id');
        attachNeighbourLinkHandlers('.cluster-member-table .satellite-id');
        attachOrbitToggleRowHandlers('.neighbour-row');
        attachOrbitToggleRowHandlers('.cluster-member-row');
    }

    function getHiddenPathIdsForMembers(members) {
        return (members || [])
            .filter((entity) => entity && !isOrbitVisible(entity))
            .map((entity) => String(entity.id));
    }

    function applyHiddenPathRows(hiddenPathIds) {
        (hiddenPathIds || []).forEach((entityId) => {
            document.querySelectorAll(`tr[data-id="${entityId}"]`).forEach((row) => {
                row.classList.add('orbit-row-hidden');
            });
        });
    }

    function snapshotUniqueModeState() {
        const searchResults = document.getElementById('searchResults');
        const searchInput = document.getElementById('searchInput');
        modeViewState.unique = {
            initialized: true,
            selectedOrbit: getSelectedOrbit(),
            showingNeighbours: uniqueModeShowingNeighbours,
            pathSnapshots: snapshotPathEntities(),
            topBottomInfoBoxHTML: topBottomInfoBox.innerHTML,
            topBottomInfoBoxDisplay: topBottomInfoBox.style.display,
            searchResultsHTML: searchResults ? searchResults.innerHTML : '',
            searchResultsDisplay: searchResults ? searchResults.style.display : 'none',
            searchInput: searchInput ? searchInput.value : '',
            camera: snapshotCamera(),
        };
    }

    function restoreUniqueModeState() {
        const state = modeViewState.unique;
        hideCompressedInfo();
        hideClusterMemberList();

        if (state.selectedOrbit) {
            setSelectedOrbit(state.selectedOrbit);
        }

        uniqueModeShowingNeighbours = !!state.showingNeighbours;
        restorePathEntities(state.pathSnapshots);

        restoreUiPanel(
            topBottomInfoBox,
            state.topBottomInfoBoxHTML || '',
            state.topBottomInfoBoxDisplay || 'none'
        );

        const searchResults = document.getElementById('searchResults');
        restoreUiPanel(
            searchResults,
            state.searchResultsHTML || '',
            state.searchResultsDisplay || 'none'
        );

        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.value = state.searchInput || '';
        }

        attachPanelLinkHandlers();
        restoreCamera(state.camera);
    }

    function snapshotClusterModeState() {
        const searchResults = document.getElementById('searchResults');
        const clusterSearchInput = document.getElementById('clusterSearchInput');
        const members =
            currentClusterLabel !== null
                ? clusterLabelToEntities.get(currentClusterLabel) || []
                : [];

        modeViewState.clusters = {
            initialized: true,
            clusterLabel: currentClusterLabel,
            highlightEntityId: currentClusterHighlightId,
            category: getSelectedClusterCategory(),
            hiddenPathIds: getHiddenPathIdsForMembers(members),
            clusterMemberListHTML: clusterMemberListBox
                ? clusterMemberListBox.innerHTML
                : '',
            clusterMemberListDisplay: clusterMemberListBox
                ? clusterMemberListBox.style.display
                : 'none',
            searchResultsHTML: searchResults ? searchResults.innerHTML : '',
            searchResultsDisplay: searchResults ? searchResults.style.display : 'none',
            searchInput: clusterSearchInput ? clusterSearchInput.value : '',
            camera: snapshotCamera(),
        };
    }

    async function restoreClusterModeState() {
        const state = modeViewState.clusters;
        hideCompressedInfo();
        hideUiPanel(topBottomInfoBox);

        if (state.category) {
            setSelectedClusterCategory(state.category);
        }

        const searchResults = document.getElementById('searchResults');
        restoreUiPanel(
            searchResults,
            state.searchResultsHTML || '',
            state.searchResultsDisplay || 'none'
        );

        const clusterSearchInput = document.getElementById('clusterSearchInput');
        if (clusterSearchInput) {
            clusterSearchInput.value = state.searchInput || '';
        }

        rebuildClusterIndex();

        if (state.clusterLabel !== null && clusterHasSyntheticPair(state.clusterLabel)) {
            const highlight = state.highlightEntityId
                ? getEntityFromId(state.highlightEntityId)
                : null;
            await displayClusterByLabel(state.clusterLabel, highlight, {
                skipFlyTo: true,
            });

            const members = clusterLabelToEntities.get(state.clusterLabel) || [];
            const hiddenSet = new Set(state.hiddenPathIds || []);
            members.forEach((entity) => {
                if (hiddenSet.has(String(entity.id)) && isOrbitVisible(entity)) {
                    removeEntityPath(entity);
                }
            });
            applyHiddenPathRows(state.hiddenPathIds);
        } else {
            restoreUiPanel(
                clusterMemberListBox,
                state.clusterMemberListHTML || '',
                state.clusterMemberListDisplay || 'none'
            );
        }

        attachPanelLinkHandlers();
        restoreCamera(state.camera);
    }

    async function handleOrbitToggle() {
        removeEntities();
        viewer.scene.requestRender();
        await delay(200);
        await showUniqueOrbits();
    }

    function enterClusteringPlaceholderView() {
        if (activeModelMode === 'unique') {
            snapshotUniqueModeState();
        }
        activeModelMode = 'clusters';

        document.body.classList.remove('orbx-mode-unique', 'orbx-mode-clusters');
        document.body.classList.add('orbx-mode-clusters');

        hideCompressedInfo();
        if (dataSource) {
            dataSource.show = true;
        }
        hideUiPanel(topBottomInfoBox);

        if (modeViewState.clusters.initialized) {
            void restoreClusterModeState();
        } else {
            removeAllEntityPaths();
            removeEntities();
            hideClusterMemberList();
            hideUiPanel(document.getElementById('searchResults'));
            rebuildClusterIndex();
            const initialCategory = getSelectedClusterCategory();
            const showInitialCluster = initialCategory
                ? pickAndShowRandomClusterForCategory(initialCategory)
                : (() => {
                    const clusterId = pickRandomClusterLabelAny();
                    return clusterId !== null
                        ? displayClusterByLabel(clusterId)
                        : Promise.resolve();
                })();
            void showInitialCluster.then(() => {
                if (currentClusterLabel !== null) {
                    snapshotClusterModeState();
                }
            });
        }

        viewer.scene.requestRender();
        console.log('[OrbX] Switched to orbital clusters view.');
    }

    function enterUniqueOrbitsView() {
        if (activeModelMode === 'clusters') {
            snapshotClusterModeState();
        }
        activeModelMode = 'unique';

        document.body.classList.remove('orbx-mode-unique', 'orbx-mode-clusters');
        document.body.classList.add('orbx-mode-unique');
        if (dataSource) {
            dataSource.show = true;
        }

        if (modeViewState.unique.initialized) {
            restoreUniqueModeState();
        } else {
            void handleOrbitToggle().then(() => {
                snapshotUniqueModeState();
            });
        }

        viewer.scene.requestRender();
        console.log('[OrbX] Switched to unique orbits view.');
    }

    function wireModelModeRadios() {
        function onModeChange() {
            if (document.getElementById('radio-mode-unique').checked) {
                enterUniqueOrbitsView();
            } else {
                enterClusteringPlaceholderView();
            }
        }
        document.querySelectorAll('input[name="modelMode"]').forEach(function(modeRadio) {
            modeRadio.addEventListener('change', onModeChange);
        });
    }

    function renderRankings(topEntities, bottomEntities) {
        const selectedOrbit = getSelectedOrbit();
        const renderTable = (title, data, indicatorClass) => {
            const rows = data.map((entity, index) => generateRankingRow(entity, index)).join("");
            return `
                <div class="rankings-card">
                    <div class="card-header">
                        <div class="header-indicator ${indicatorClass}"></div>
                        <h2 class="card-title">${title}</h2>
                    </div>
                    <table class="rankings-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Score</th>
                                <th>NORAD ID</th>
                                <th>Satellite Name</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                    <div class="table-footer">
                        ${title.includes('Most')
                            ? 'Higher score indicates more unique orbital characteristics'
                            : 'Lower score indicates more common orbital characteristics'
                        }
                    </div>
                </div>
            `;
        };
    
        return `
            <div class="container">
                ${renderTable(`5 Most Unique Orbits (${selectedOrbit})`, topEntities, 'red-indicator')}
                ${renderTable(`5 Least Unique Orbits (${selectedOrbit})`, bottomEntities, 'green-indicator')}
            </div>
        `;
    }
    
    // Example function to update the topBottomInfoBox content
    function updateRankingsDisplay(topEntities, bottomEntities) {
        hideClusterMemberList();
        hideUiPanel(document.getElementById('searchResults'));
        showUiPanel(topBottomInfoBox, renderRankings(topEntities, bottomEntities));
        attachNeighbourLinkHandlers('.satellite-id');
    }

    function generateNeighbourSatelliteList(satellites) {
        const rows = satellites.list.map((sat, index) => generateNeighbourRow(sat, index)).join("");

        let html = `
        <div class="neighbour-list-container">
          <div class="neighbour-list-rankings-card">
            <div class="neighbour-list-card-header">
              <h2 class="neighbour-list-target-satellite">
              10 Nearest Satellites for NORAD ID: 
              <span class="neighbour-list-target-badge">${satellites.targetId}</span>
              </h2>
            </div>
            <table class="neighbour-list-rankings-table">
              <thead>
                <tr>
                  <th>Score</th>
                  <th>NORAD ID</th>
                  <th>Satellite Name</th>
                </tr>
              </thead>
              <tbody>`;
        
        satellites.list.forEach((sat, index) => {
            const displayNorad = formatDisplayNoradFromId(sat.id);
            html += `
                <tr class="neighbour-row" data-id="${sat.id}">
                    <td>${index + 1}</td>
                    <td>
                    <a href="#" class="satellite-id" data-id="${sat.id}">${displayNorad}</a>
                    </td>
                    <td class="neighbour-list-sat-name">${sat.name}</td>
                </tr>`;
        });
        
        html += `
              </tbody>
            </table>
            <div class="neighbour-list-table-footer">
              <div class="neighbour-list-legend">
                <div class="neighbour-list-legend-item">
                  <span class="neighbour-list-color-indicator neighbour-list-color-blue"></span>
                  <span>Searched satellite</span>
                </div>
                <div class="neighbour-list-legend-item">
                  <span class="neighbour-list-color-indicator neighbour-list-color-yellow"></span>
                  <span>Nearby satellites</span>
                </div>
              </div>
            </div>
          </div>
        </div>`;
        
        return html;
    }

    if (typeof openNav === 'function') {
        openNav();
    }
});

function wireViewerResize(viewer) {
    const resize = () => viewer.resize();
    window.addEventListener('resize', resize);
    requestAnimationFrame(() => requestAnimationFrame(resize));
}

/**
 * Laptop trackpads usually expose pinch as ctrl+wheel (Chrome/Edge/Firefox)
 * or Safari gesture* events — not Cesium's touch PINCH. Map those to camera zoom
 * so pinch/spread works more like Google Earth.
 */
function wireTrackpadPinchZoom(viewer) {
    const canvas = viewer.scene.canvas;
    if (!canvas) return;

    const zoomFromDelta = (deltaY) => {
        const camera = viewer.camera;
        const cartographic = camera.positionCartographic;
        const height = cartographic
            ? Math.max(cartographic.height, 1.0)
            : Math.max(Cesium.Cartesian3.magnitude(camera.position), 1.0);
        // Scale with altitude so pinch feels consistent from LEO to global views.
        const amount = Math.abs(deltaY) * height * 0.0025;
        if (amount <= 0) return;
        // Browser convention: positive deltaY = pinch-in = zoom out.
        if (deltaY > 0) {
            camera.zoomOut(amount);
        } else {
            camera.zoomIn(amount);
        }
        viewer.scene.requestRender();
    };

    canvas.addEventListener(
        'wheel',
        (event) => {
            // Trackpad pinch is reported as a wheel event with ctrlKey set.
            if (!event.ctrlKey && !event.metaKey) return;
            event.preventDefault();
            event.stopPropagation();
            zoomFromDelta(event.deltaY);
        },
        { passive: false, capture: true }
    );

    // Safari (and some WebKit builds) use gesture events for trackpad pinch.
    let lastGestureScale = 1;
    const onGestureStart = (event) => {
        event.preventDefault();
        lastGestureScale = event.scale || 1;
    };
    const onGestureChange = (event) => {
        event.preventDefault();
        const scale = event.scale || 1;
        const ratio = scale / (lastGestureScale || 1);
        lastGestureScale = scale;
        if (!Number.isFinite(ratio) || ratio === 1) return;
        // Convert scale ratio to a wheel-like delta: spread (ratio>1) → zoom in.
        const deltaY = (1 - ratio) * 120;
        zoomFromDelta(deltaY);
    };
    canvas.addEventListener('gesturestart', onGestureStart, { passive: false });
    canvas.addEventListener('gesturechange', onGestureChange, { passive: false });
    canvas.addEventListener('gestureend', (event) => event.preventDefault(), {
        passive: false,
    });
}