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
        geocoder: false,
        sceneModePicker: false,
        baseLayerPicker: false,
        navigationHelpButton: false,
        homeButton: false
    });

    viewer.scene.globe.enableLighting = true;
    viewer.scene.sun = new Cesium.Sun();
    viewer.scene.moon = new Cesium.Moon();
    const topBottomInfoBox = document.getElementById('topBottomInfoBox');
    const clusterMemberListBox = document.getElementById('clusterMemberListBox');
    document.body.classList.add('orbx-mode-unique');
    wireModelModeRadios();

    // on load or refresh, clear the search bars
    document.getElementById('searchInput').value = '';
    const clusterSearchInputEl = document.getElementById('clusterSearchInput');
    if (clusterSearchInputEl) {
        clusterSearchInputEl.value = '';
    }

    let dataSource;
    let highlightedEntities = [];
    const clusterLabelToEntities = new Map();
    try {
        const latestAsset = await fetchLatestAsset();
        const assetId = latestAsset.id;
        
        const resource = await Cesium.IonResource.fromAssetId(assetId);
        dataSource = await Cesium.CzmlDataSource.load(resource);
        await viewer.dataSources.add(dataSource);
        viewer.clock.currentTime = Cesium.JulianDate.now();
        viewer.clock.multiplier = 50;

        const step = 10;

        const animationViewModel = viewer.animation.viewModel;
        animationViewModel.playForwardViewModel.command.beforeExecute.addEventListener(function(commandInfo) {
            viewer.clock.multiplier += step;
        });

        animationViewModel.playReverseViewModel.command.beforeExecute.addEventListener(function(commandInfo) {
            viewer.clock.multiplier -= step;
        });

        loadingScreen.style.display = 'none';

        const urlParams = new URLSearchParams(window.location.search);
        const idFromURL = urlParams.get('id');
        if (idFromURL) {
            performSearch(idFromURL);
        }

        dataSource.entities.values.forEach(entity => entity.show = false);

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
            const uniqueness = entity.properties.uniqueness?.getValue(now);
            const uniquenessStr = (typeof uniqueness === 'number')
                ? (uniqueness < 0.01 ? uniqueness.toExponential(2) : uniqueness.toFixed(2))
                : "N/A";
            let clusterLine = '';
            if (document.body.classList.contains('orbx-mode-clusters')) {
                const cl = getClusterLabelFromEntity(entity, now);
                if (cl !== null && cl !== -1) {
                    clusterLine = `<strong>Cluster ID:</strong> ${cl} <br>`;
                }
            }
            infoBox.innerHTML = `<div style="padding: 5px 10px; white-space: nowrap;">
                    <strong>NORAD ID:</strong> ${entity.id} <br>
                    <strong>Name:</strong> ${entity.name || "N/A"} <br>
                    <strong>Uniqueness:</strong> ${uniquenessStr} <br>
                    ${clusterLine}
                </div>`;
        } else {
            infoBox.innerHTML = `<div style="padding: 5px 10px; white-space: nowrap;">Entity ID: ${entityId}</div>`;
        }
        
        // Ensure the infoBox resizes to fit its content.
        infoBox.style.display = 'inline-block';
        infoBox.style.position = 'absolute';
        infoBox.style.fontSize = '12px';
        infoBox.style.width = '10%';
        infoBox.style.zIndex = '9999'; // Bring the info box to the front
    
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

    // initialise the model
    removeEntities();
    document.getElementById('radio-leo').checked = true;
    handleOrbitToggle();

    // Define toggleOrbit to show/hide the orbit path.
    function toggleOrbit(entityId, color) {
        const entity = dataSource && dataSource.entities && dataSource.entities.getById 
            ? dataSource.entities.getById(entityId)
            : null;
        if (!entity) return;
        if (entity.path) {
            removeEntityPath(entity);
        } else {
            showEntityPath(entity, color);
        }
    }
    window.toggleOrbit = toggleOrbit;

    function showEntityPath(entity, color=undefined) {
        // Use the passed color, or the saved color, otherwise default to white.
        const orbit_color = color || entity.orbitColor || Cesium.Color.WHITE;
        // Store the color on the entity for later toggling.
        entity.orbitColor = orbit_color;
        
        // Create or update the entity path with the correct color.
        if (entity.path) {
            entity.path.material = new Cesium.ColorMaterialProperty(orbit_color);
            entity.path.width = 2;
            entity.path.show = true;
        } else {
            entity.path = new Cesium.PathGraphics({
                show: true,
                material: new Cesium.ColorMaterialProperty(orbit_color),
                width: 2
            });
        }
        
        if (!viewer.entities.contains(entity)) {
            viewer.entities.add(entity);
        }
        entity.show = true;
    }

    function removeEntityPath(entity) {
        if (entity.path) {
            entity.path = undefined;
            viewer.entities.remove(entity);
        }
    }

    function removeAllEntityPaths() {
        if (!dataSource || !dataSource.entities) {
            highlightedEntities = [];
            return;
        }
        dataSource.entities.values.forEach(entity => {
            if (entity.path) {
                entity.path = undefined;    // remove path
                viewer.entities.remove(entity); // remove entity from viewer
            }
        });
        highlightedEntities = [];
    }

    function removeEntities() {
        // Clear any manually added orbit paths
        viewer.entities.removeAll();

        // Also hide dataSource entities if needed
        if (!dataSource || !dataSource.entities) {
            return;
        }
        dataSource.entities.values.forEach(entity => entity.show = false);
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

    function isRealSatelliteEntity(entity, time) {
        return !!entity && !isSyntheticEntity(entity, time);
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

    function rebuildClusterIndex() {
        clusterLabelToEntities.clear();
        if (!dataSource || !dataSource.entities) return;
        const now = Cesium.JulianDate.now();
        dataSource.entities.values.forEach((entity) => {
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
        if (candidates.length === 0) return null;
        return candidates[Math.floor(Math.random() * candidates.length)];
    }

    /** Random cluster with both synthetic orbits; excludes noise (-1). */
    function pickRandomClusterLabelAny() {
        const candidates = [];
        clusterLabelToEntities.forEach((entities, label) => {
            if (label === -1) return;
            if (!clusterHasSyntheticPair(label)) return;
            candidates.push(label);
        });
        if (candidates.length === 0) return null;
        return candidates[Math.floor(Math.random() * candidates.length)];
    }

    const SYNTHETIC_PATH_COLOR_FRECHET = Cesium.Color.RED;
    const SYNTHETIC_PATH_COLOR_MAX_SEPARATION = Cesium.Color.BLUE;

    function getPathColorForClusterEntity(entity, highlightEntity) {
        const now = Cesium.JulianDate.now();
        const synthType = getSyntheticTypeFromEntity(entity, now);
        if (synthType === 'frechet') return SYNTHETIC_PATH_COLOR_FRECHET;
        if (synthType === 'max_separation') return SYNTHETIC_PATH_COLOR_MAX_SEPARATION;

        const memberColor = Cesium.Color.fromCssColorString('#20c997');
        const hl =
            highlightEntity && String(entity.id) === String(highlightEntity.id);
        return hl ? Cesium.Color.fromCssColorString('#00bfff') : memberColor;
    }

    async function displayClusterByLabel(clusterLabel, highlightEntity) {
        rebuildClusterIndex();
        if (!clusterHasSyntheticPair(clusterLabel)) {
            console.warn('[OrbX] Cluster has no synthetic orbit pair:', clusterLabel);
            return;
        }
        removeAllEntityPaths();
        removeEntities();
        const members = clusterLabelToEntities.get(clusterLabel);
        if (!members || members.length === 0) {
            console.warn('[OrbX] No entities for cluster label', clusterLabel);
            return;
        }
        members.forEach((entity) => {
            showEntityPath(entity, getPathColorForClusterEntity(entity, highlightEntity));
        });
        displayClusterMemberList(clusterLabel, members, highlightEntity);
        await viewer.flyTo(members, {
            duration: 2,
            offset: new Cesium.HeadingPitchRange(
                Cesium.Math.toRadians(0),
                Cesium.Math.toRadians(-90)
            )
        });
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
        const sr = document.getElementById('searchResults');
        if (sr) {
            const members = clusterLabelToEntities.get(clusterId) || [];
            const realN = getClusterRealMemberCount(members);
            const tier = clusterSizeTier(realN);
            const tierText = tier
                ? tierBandLabel(tier)
                : `Size outside micro–mega bands (n=${realN} real)`;
            sr.innerHTML =
                `<div style="padding:12px;background:rgba(30,30,30,0.85);color:#eee;border-radius:8px;max-width:420px;font-family:Arial,sans-serif;font-size:14px;">` +
                `<strong>Cluster ${clusterId}</strong><br>` +
                `${realN} satellites (+ synthetics) · ${tierText}</div>`;
            sr.style.display = 'block';
        }
    }

    function getOrbitEntities(selectedOrbit){
        const entities = dataSource.entities.values;

        // console.log("getOrbitEntities called with selectedOrbit: ", selectedOrbit);

        const orbitEntities = entities.filter((entity) => {
            if (!isRealSatelliteEntity(entity)) return false;
            const orbit_class = entity.properties.orbit_class?.getValue();
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
        entities.sort((a, b) => a.properties.rank?.getValue() - b.properties.rank?.getValue());

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
    
        // Zoom in on the displayed satellites
        await viewer.flyTo(
            [...topEntities, ...bottomEntities],
            {
                duration: 1,
                offset: new Cesium.HeadingPitchRange(
                    Cesium.Math.toRadians(0),
                    Cesium.Math.toRadians(-90),
                )
            }
        );

        updateRankingsDisplay(topEntities, bottomEntities);
    }

    // if there is a change in any of the orbit filter radios
    ['radio-leo', 'radio-meo', 'radio-geo', 'radio-heo'].forEach(id => {
        const radio = document.getElementById(id);
        if (radio) {
            radio.addEventListener('change', function() {
                if (!document.getElementById('radio-mode-unique').checked) {
                    return;
                }
                console.log("radio change event");
                removeEntities();
                handleOrbitToggle();
            });
        }
    });

    function getSelectedClusterCategory() {
        if (document.getElementById('radio-micro').checked) return 'micro';
        if (document.getElementById('radio-minor').checked) return 'minor';
        if (document.getElementById('radio-major').checked) return 'major';
        if (document.getElementById('radio-mega').checked) return 'mega';
        return 'micro';
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
                    isRealSatelliteEntity(e)
                );
                if (realEntities.length === 0) {
                    alert('No real satellites available in the data source.');
                    return;
                }
                const randomIndex = Math.floor(Math.random() * realEntities.length);
                searchId = realEntities[randomIndex].id;
            }

            const radios = ['radio-leo', 'radio-meo', 'radio-geo', 'radio-heo'];
            radios.forEach(radio => {
                document.getElementById(radio).checked = false;
            });

            const searchedEntity = dataSource.entities.getById(searchId);
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

            const neighbourIds = searchedEntity.properties.neighbours?.getValue();
            console.log("neighbourIds: ", neighbourIds);

            const neighbourEntities = [];
            if (neighbourIds) {
                const neighbourIdArray = Object.values(neighbourIds);
                neighbourIdArray.forEach(neighbourId => {
                    const neighbourEntity = dataSource.entities.getById(neighbourId);
                    if (neighbourEntity && isRealSatelliteEntity(neighbourEntity)) {
                        neighbourEntities.push(neighbourEntity);
                    }
                });
            }

            const searchResults = document.getElementById('searchResults');
            topBottomInfoBox.style.display = 'none';
            hideClusterMemberList();

            if (!neighbourEntities || neighbourEntities.length === 0) {
                console.log("No neighbours found for NORAD ID: " + searchId);
                if (searchResults) {
                    searchResults.innerHTML = `<p>No neighbours found for NORAD ID: ${searchId}</p>`;
                    searchResults.style.display = 'block';
                }
                return;
            }

            if (searchResults) {
                console.log("searchResults found");
                searchResults.innerHTML = `<h3>10 Nearest Satellites for NORAD ID: ${searchId}</h3>` +
                    generateNeighbourSatelliteList({ targetId: searchId, list: neighbourEntities });
                searchResults.style.display = 'block';
                attachNeighbourLinkHandlers('.neighbour-list-container .satellite-id');
                attachOrbitToggleRowHandlers('.neighbour-row');
            }

            removeAllEntityPaths();
            removeEntities();

            neighbourEntities.forEach(neighbour => showEntityPath(neighbour, Cesium.Color.YELLOW));
            showEntityPath(searchedEntity, Cesium.Color.BLUE);

            await viewer.flyTo(
                [...neighbourEntities, searchedEntity],
                {
                    duration: 2,
                    offset: new Cesium.HeadingPitchRange(
                        Cesium.Math.toRadians(0),
                        Cesium.Math.toRadians(-90)
                    )
                }
            );

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
                const sr = document.getElementById('searchResults');
                if (sr) {
                    const members = clusterLabelToEntities.get(clusterId) || [];
                    const realN = getClusterRealMemberCount(members);
                    const tier = clusterSizeTier(realN);
                    const tierText = tier
                        ? tierBandLabel(tier)
                        : `Size outside micro–mega bands (n=${realN} real)`;
                    sr.innerHTML =
                        `<div style="padding:12px;background:rgba(30,30,30,0.85);color:#eee;border-radius:8px;max-width:420px;font-family:Arial,sans-serif;font-size:14px;">` +
                        `<strong>Random cluster ${clusterId}</strong> <span style="opacity:0.85">(synthetic pair required)</span><br>` +
                        `${realN} satellites (+ synthetics) · ${tierText}</div>`;
                    sr.style.display = 'block';
                }
                return;
            }

            const searchedEntity = dataSource.entities.getById(searchId);
            if (!searchedEntity) {
                alert("NORAD ID not found in data source");
                return;
            }

            const now = Cesium.JulianDate.now();
            const lab = getClusterLabelFromEntity(searchedEntity, now);

            const searchResults = document.getElementById('searchResults');
            if (clusterMemberListBox) clusterMemberListBox.style.display = 'none';

            removeAllEntityPaths();
            removeEntities();

            if (lab === null || lab === -1) {
                if (searchResults) {
                    searchResults.innerHTML =
                        `<div style="padding:12px;background:rgba(30,30,30,0.85);color:#eee;border-radius:8px;max-width:420px;font-family:Arial,sans-serif;font-size:14px;">` +
                        `<strong>NORAD ${searchId}</strong><br>` +
                        `Noise / unclustered — no cluster assignment (not shown).</div>`;
                    searchResults.style.display = 'block';
                }

                const neighbourIds = searchedEntity.properties.neighbours?.getValue();
                const neighbourEntities = [];
                if (neighbourIds) {
                    Object.values(neighbourIds).forEach(neighbourId => {
                        const ne = dataSource.entities.getById(neighbourId);
                        if (ne) neighbourEntities.push(ne);
                    });
                }

                if (neighbourEntities.length === 0) {
                    showEntityPath(searchedEntity, Cesium.Color.BLUE);
                    await viewer.flyTo(searchedEntity, {
                        duration: 2,
                        offset: new Cesium.HeadingPitchRange(
                            Cesium.Math.toRadians(0),
                            Cesium.Math.toRadians(-90)
                        )
                    });
                    return;
                }

                if (searchResults) {
                    searchResults.innerHTML +=
                        `<h3 style="color:#eee;margin-top:14px;">10 nearest neighbours</h3>` +
                        generateNeighbourSatelliteList({ targetId: searchId, list: neighbourEntities });
                    attachNeighbourLinkHandlers('.neighbour-list-container .satellite-id');
                    attachOrbitToggleRowHandlers('.neighbour-row');
                }

                neighbourEntities.forEach(neighbour =>
                    showEntityPath(neighbour, Cesium.Color.YELLOW));
                showEntityPath(searchedEntity, Cesium.Color.BLUE);

                await viewer.flyTo(
                    [...neighbourEntities, searchedEntity],
                    {
                        duration: 2,
                        offset: new Cesium.HeadingPitchRange(
                            Cesium.Math.toRadians(0),
                            Cesium.Math.toRadians(-90)
                        )
                    }
                );
                return;
            }

            if (!clusterHasSyntheticPair(lab)) {
                if (searchResults) {
                    searchResults.innerHTML =
                        `<div style="padding:12px;background:rgba(30,30,30,0.85);color:#eee;border-radius:8px;max-width:420px;font-family:Arial,sans-serif;font-size:14px;">` +
                        `<strong>NORAD ${searchId}</strong><br>` +
                        `Cluster ID: <strong>${lab}</strong><br>` +
                        `This cluster has no synthetic orbits in the dataset (only ~1% of clusters include Fréchet + max-separation).` +
                        `</div>`;
                    searchResults.style.display = 'block';
                }
                showEntityPath(searchedEntity, Cesium.Color.BLUE);
                await viewer.flyTo(searchedEntity, {
                    duration: 2,
                    offset: new Cesium.HeadingPitchRange(
                        Cesium.Math.toRadians(0),
                        Cesium.Math.toRadians(-90)
                    )
                });
                return;
            }

            const members = clusterLabelToEntities.get(lab) || [];
            const realN = getClusterRealMemberCount(members);
            const totalN = members.length;
            const tier = clusterSizeTier(realN);

            if (searchResults) {
                searchResults.innerHTML =
                    `<div style="padding:12px;background:rgba(30,30,30,0.85);color:#eee;border-radius:8px;max-width:420px;font-family:Arial,sans-serif;font-size:14px;">` +
                    `<strong>NORAD ${searchId}</strong><br>` +
                    `Cluster ID: <strong>${lab}</strong><br>` +
                    `${realN} real satellites (${totalN} entities including synthetics)<br>` +
                    `${tier ? tierBandLabel(tier) : 'Band: outside standard tiers (size ' + realN + ' real)'}` +
                    `</div>`;
                searchResults.style.display = 'block';
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
                    clusterFakePlaceholder.textContent = placeholders[clusterPlaceholderIndex];
                    clusterPlaceholderIndex = (clusterPlaceholderIndex + 1) % placeholders.length;
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
        const entity = dataSource && dataSource.entities && typeof dataSource.entities.getById === 'function'
            ? dataSource.entities.getById(entityId)
            : null;

        return entity;
    }

    function hideClusterMemberList() {
        if (clusterMemberListBox) {
            clusterMemberListBox.style.display = 'none';
            clusterMemberListBox.innerHTML = '';
        }
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
        let roleLabel = 'Member';
        let roleClass = 'cluster-role-member';
        if (synthType === 'frechet') {
            roleLabel = 'Fréchet';
            roleClass = 'cluster-role-frechet';
        } else if (synthType === 'max_separation') {
            roleLabel = 'Max-separation';
            roleClass = 'cluster-role-maxsep';
        }
        const isHighlight =
            highlightEntity && String(entity.id) === String(highlightEntity.id);
        const rowClass = isHighlight
            ? 'cluster-member-row cluster-member-row-highlight neighbour-row'
            : 'cluster-member-row neighbour-row';

        return `
            <tr class="${rowClass}" data-id="${entity.id}">
                <td>${index + 1}</td>
                <td class="${roleClass}">${roleLabel}</td>
                <td><a href="#" class="satellite-id" data-id="${entity.id}">${entity.id}</a></td>
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
        const rows = sorted
            .map((entity, index) => generateClusterMemberRow(entity, index, highlightEntity))
            .join('');

        return `
            <div class="container">
                <div class="rankings-card cluster-member-card">
                    <div class="card-header">
                        <h2 class="card-title">Cluster ${clusterLabel} · ${tierText}</h2>
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
                                Fréchet synthetic
                            </span>
                            <span class="cluster-legend-item">
                                <span class="header-indicator blue-indicator"></span>
                                Max-separation synthetic
                            </span>
                            <span class="cluster-legend-item">
                                <span class="header-indicator teal-indicator"></span>
                                Cluster member
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function displayClusterMemberList(clusterLabel, members, highlightEntity) {
        if (!clusterMemberListBox || !members || members.length === 0) return;

        const searchResults = document.getElementById('searchResults');
        if (searchResults) {
            searchResults.style.display = 'none';
        }

        clusterMemberListBox.innerHTML = renderClusterMemberList(
            clusterLabel,
            members,
            highlightEntity
        );
        clusterMemberListBox.style.display = 'block';
        attachNeighbourLinkHandlers('.cluster-member-table .satellite-id');
        attachOrbitToggleRowHandlers('.cluster-member-row');
    }

    function generateSatelliteList(satellites) {
        return `<ul style="padding-left: 20px; list-style-type: none;">
            ${satellites.map(satellite => {
                const uniqueness = satellite.properties.uniqueness?.getValue();
                const uniquenessStr = (typeof uniqueness === 'number')
                    ? (uniqueness < 0.01 ? uniqueness.toExponential(2) : uniqueness.toFixed(2))
                    : "N/A";
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
        // close the search results panel
        const searchResults = document.getElementById('searchResults');
        searchResults.style.display = 'none';
        
        const selectedOrbit = getSelectedOrbit();
        // get the top and bottom 5 entities
        const entities = getOrbitEntities(selectedOrbit);
        const [topEntities, bottomEntities] = getTopBottomEntities(entities);
    
        // Build the info box content using generateSatelliteList.
        let infoboxContent = `<h3><span class="box red"></span>5 Most Unique Orbits (${selectedOrbit})</h3>` + generateSatelliteList(topEntities);
        infoboxContent += `<h3><span class="box green"></span>5 Least Unique Orbits (${selectedOrbit})</h3>` + generateSatelliteList(bottomEntities);
    
        const topBottomInfoBox = document.getElementById('topBottomInfoBox');
        topBottomInfoBox.innerHTML = infoboxContent;
        
        
        topBottomInfoBox.style.display = 'block';

        attachNeighbourLinkHandlers('.satellite-id');
    }

    function generateRankingRow(satellite, index) {
        const uniqueness = satellite.properties.uniqueness?.getValue();
        const uniquenessStr = (typeof uniqueness === 'number')
            ? (uniqueness < 0.01 ? uniqueness.toExponential(2) : uniqueness.toFixed(2))
            : "N/A";
        return `
            <tr>
                <td>${index + 1}</td>
                <td class="score-cell">${uniquenessStr}</td>
                <td><a href="#" class="satellite-id" data-id="${satellite.id}">${satellite.id}</a></td>
                <td class="sat-name">${satellite.name || "N/A"}</td>
            </tr>
        `;
    }
    
    function generateNeighbourRow(satellite, index) {
        return `
            <tr class="neighbour-row" data-id="${satellite.id}">
                <td>${index + 1}</td>
                <td><a href="#" class="satellite-id" data-id="${satellite.id}">${satellite.id}</a></td>
                <td class="neighbour-list-sat-name">${satellite.name}</td>
            </tr>
        `;
    }

    function handleOrbitToggle() {
        // console.log("handleOrbitToggle called");
        removeEntities();
        showUniqueOrbits();
        displayUniqueOrbitList();
        //clear entities
        
    }

    function enterClusteringPlaceholderView() {
        // 1) Body mode class first so all CSS-tied chrome (side nav, rankings, search panels) updates in one reflow.
        document.body.classList.remove('orbx-mode-unique', 'orbx-mode-clusters');
        document.body.classList.add('orbx-mode-clusters');

        hideCompressedInfo();
        removeAllEntityPaths();
        removeEntities();
        if (dataSource) {
            dataSource.show = true;
        }
        const sr = document.getElementById('searchResults');
        topBottomInfoBox.style.display = 'none';
        hideClusterMemberList();
        if (sr) {
            sr.style.display = 'none';
        }
        rebuildClusterIndex();
        void pickAndShowRandomClusterForCategory(getSelectedClusterCategory());
        viewer.scene.requestRender();
        console.log('[OrbX] Switched to orbital clusters view.');
    }

    function enterUniqueOrbitsView() {
        document.body.classList.remove('orbx-mode-unique', 'orbx-mode-clusters');
        document.body.classList.add('orbx-mode-unique');
        if (dataSource) {
            dataSource.show = true;
        }
        handleOrbitToggle();
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
        const topBottomInfoBox = document.getElementById('topBottomInfoBox');
        topBottomInfoBox.innerHTML = renderRankings(topEntities, bottomEntities);
        topBottomInfoBox.style.display = 'block';
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
            html += `
                <tr class="neighbour-row" data-id="${sat.id}">
                    <td>${index + 1}</td>
                    <td>
                    <a href="#" class="satellite-id" data-id="${sat.id}">${sat.id}</a>
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