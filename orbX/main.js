document.addEventListener("DOMContentLoaded", async function () {
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
        baseLayerPicker: true,
        navigationHelpButton: true,
        homeButton: true
    });
    if (CONFIG.DEBUG === true) {
        window.viewer = viewer;
    }
    
    viewer.scene.globe.enableLighting = true;
    viewer.scene.sun = new Cesium.Sun();
    viewer.scene.moon = new Cesium.Moon();
    
    let dataSource;
    try {
        // const latestAsset = await fetchLatestAsset();
        // const assetId = latestAsset.id;
        // const resource = await Cesium.IonResource.fromAssetId(assetId);
        // dataSource = await Cesium.CzmlDataSource.load(resource);
        dataSource = await Cesium.CzmlDataSource.load('output.czml');
        await viewer.dataSources.add(dataSource);
        window.czmlDataSource = dataSource;
        viewer.clock.currentTime = Cesium.JulianDate.now();
        viewer.clock.multiplier = 50;
  
        const step = 10;
        const animationViewModel = viewer.animation.viewModel;
        animationViewModel.playForwardViewModel.command.beforeExecute.addEventListener(function() {
            viewer.clock.multiplier += step;
        });
        animationViewModel.playReverseViewModel.command.beforeExecute.addEventListener(function() {
            viewer.clock.multiplier -= step;
        });
  
        loadingScreen.style.display = 'none';
        addLegend();
        addDatasetChecklist();
  
        const urlParams = new URLSearchParams(window.location.search);
        const idFromURL = urlParams.get('id');
        if (idFromURL) {
            performSearch(idFromURL);
        }
  
        dataSource.entities.values.forEach(entity => {
            entity.show = true;
            if (!entity.path) {
                entity.show = true;
                showEntityPath(entity);
            } else {
                entity.path.show = true;
            }
        });

        // ===== Hover infoBox support =====
        const infoBox = document.getElementById("infoBox");

        function showCompressedInfo(entityData, mousePosition) {
            const entityId = (typeof entityData === 'object' && entityData.id)
                ? entityData.id
                : entityData;

            const entity = dataSource && dataSource.entities && dataSource.entities.getById
                ? dataSource.entities.getById(entityId)
                : null;

            const now = viewer.clock.currentTime;
            const offset = 10;

            if (entity) {
                infoBox.innerHTML = `<div style="padding: 5px 10px; white-space: nowrap;">
                        <strong>Name:</strong> ${entity.name} <br>
                        <strong>NORAD ID:</strong> ${entity.satNo} <br>
                        <strong>Apogee:</strong> ${entity.properties.apogee} km <br>
                        <strong>Inclination: </strong> ${entity.properties.inclination}° <br>
                    </div>`;
            } else {
                infoBox.innerHTML = `<div style="padding: 5px 10px; white-space: nowrap;">Entity ID: ${entityId}</div>`;
            }

            infoBox.style.display = 'inline-block';
            infoBox.style.position = 'absolute';
            infoBox.style.fontSize = '12px';
            infoBox.style.width = '10%';
            infoBox.style.zIndex = '9999';

            infoBox.style.left = (mousePosition.x + offset) + 'px';
            infoBox.style.top = (mousePosition.y + offset) + 'px';

            const boxRect = infoBox.getBoundingClientRect();

            if (boxRect.right > window.innerWidth) {
                infoBox.style.left = (mousePosition.x - boxRect.width - offset) + 'px';
            }
            if (boxRect.bottom > window.innerHeight) {
                infoBox.style.top = (mousePosition.y - boxRect.height - offset) + 'px';
            }
            if (boxRect.top < 0) {
                infoBox.style.top = (mousePosition.y + offset) + 'px';
            }
        }

        function hideCompressedInfo() {
            infoBox.style.display = 'none';
            infoBox.innerHTML = '';
        }

        viewer.screenSpaceEventHandler.setInputAction(function onMouseMove(movement) {
            const pickedObject = viewer.scene.pick(movement.endPosition);
            if (Cesium.defined(pickedObject) && pickedObject.id) {
                showCompressedInfo(pickedObject.id, movement.endPosition);
            } else {
                hideCompressedInfo();
            }
        }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
        // ===== end hover support =====

    } catch (error) {
        console.log(error);
    }
    
    function addLegend() {
        const legend = document.createElement('div');
        legend.style.position = 'absolute';
        legend.style.top = '10px';
        legend.style.left = '10px';
        legend.style.background = 'rgba(0,0,0,0.6)';
        legend.style.color = '#fff';
        legend.style.padding = '8px 10px';
        legend.style.borderRadius = '4px';
        legend.style.fontFamily = 'sans-serif';
        legend.style.fontSize = '12px';
        legend.style.zIndex = '1000';

        const row = (colorCss, label) => {
            const item = document.createElement('div');
            item.style.display = 'flex';
            item.style.alignItems = 'center';
            item.style.margin = '4px 0';
            const swatch = document.createElement('span');
            swatch.style.display = 'inline-block';
            swatch.style.width = '12px';
            swatch.style.height = '12px';
            swatch.style.marginRight = '8px';
            swatch.style.border = '1px solid #ddd';
            swatch.style.background = colorCss;
            const text = document.createElement('span');
            text.textContent = label;
            item.appendChild(swatch);
            item.appendChild(text);
            return item;
        };

        legend.appendChild(row('#ffffff', 'Optimum orbit'));
        legend.appendChild(row('#00ff00', 'Input TLE'));

        document.body.appendChild(legend);
    }

    function addDatasetChecklist() {
        const container = document.createElement('div');
        container.style.position = 'absolute';
        container.style.top = '70px';
        container.style.left = '10px';
        container.style.background = 'rgba(0,0,0,0.6)';
        container.style.color = '#fff';
        container.style.padding = '8px 10px';
        container.style.borderRadius = '4px';
        container.style.fontFamily = 'sans-serif';
        container.style.fontSize = '12px';
        container.style.zIndex = '1000';
        container.style.maxHeight = '40vh';
        container.style.overflowY = 'auto';

        const title = document.createElement('div');
        title.textContent = 'Datasets';
        title.style.marginBottom = '6px';
        title.style.fontWeight = 'bold';
        container.appendChild(title);

        const dsSet = new Set();
        dataSource.entities.values.forEach(entity => {
            const dsProp = entity.properties && entity.properties.dataset;
            if (dsProp) {
                const val = dsProp.getValue(viewer.clock.currentTime);
                if (val) dsSet.add(val);
            }
        });

        const datasets = Array.from(dsSet).sort();
        const state = {};

        const applyFilter = () => {
            dataSource.entities.values.forEach(entity => {
                const dsProp = entity.properties && entity.properties.dataset;
                const dsVal = dsProp ? dsProp.getValue(viewer.clock.currentTime) : undefined;
                if (!dsVal || state[dsVal] === undefined) {
                    entity.show = true;
                } else {
                    entity.show = !!state[dsVal];
                }
            });
        };

        datasets.forEach(ds => {
            state[ds] = true;
            const row = document.createElement('label');
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.gap = '6px';
            row.style.margin = '4px 0';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = true;
            cb.addEventListener('change', () => {
                state[ds] = cb.checked;
                applyFilter();
            });
            const span = document.createElement('span');
            span.textContent = ds;
            row.appendChild(cb);
            row.appendChild(span);
            container.appendChild(row);
        });

        document.body.appendChild(container);
    }

    function showEntityPath(entity) {
        console.log('showEntityPath called for entity:', entity.id || entity.name);
        
        let orbit_color = Cesium.Color.WHITE;

        const currentTime = viewer.clock.currentTime;
        let satNoValue = undefined;
        if (entity.properties && entity.properties.satNo) {
            satNoValue = entity.properties.satNo.getValue(currentTime);
        }
        if (satNoValue === '99999') {
            console.log('satNo is 99999, setting to white');
            orbit_color = Cesium.Color.WHITE;
        } else {
            orbit_color = Cesium.Color.GREEN;
        }
    
        console.log('orbit_color:', orbit_color);
    
        if (!entity.path) {
            entity.path = new Cesium.PathGraphics({
                show: true,
                material: new Cesium.ColorMaterialProperty(orbit_color),
                width: 2
            });
            console.log('Created new path with orbit_color:', orbit_color);
        } else {
            entity.path.material = new Cesium.ColorMaterialProperty(orbit_color);
            entity.path.width = 2;
            entity.path.show = true;
            console.log('Updated existing path with orbit_color:', orbit_color);
        }
    
        if (!viewer.entities.contains(entity)) {
            viewer.entities.add(entity);
            console.log('Entity added to viewer:', entity.id || entity.name);
        }
        entity.show = true;
    }
});
