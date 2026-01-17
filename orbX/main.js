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

    // on load or refresh, clear the search bar
    // document.getElementById('searchInput').value = '';

    let dataSource;
    let highlightedEntities = [];
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

        // dataSource.entities.values.forEach(entity => entity.show = false);

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
            
            infoBox.innerHTML = `<div style="padding: 5px 10px; white-space: nowrap;">
                    <strong>Name:</strong> ${entity.name || "N/A"} <br>
                    <strong>NORAD ID:</strong> ${entity.id} <br>
                    <strong>Apogee:</strong> ${entity.properties.apogee ? entity.properties.apogee.getValue(now).toFixed(2) : "N/A"} km <br>
                    <strong>Inclination: </strong> ${entity.properties.inclination ? entity.properties.inclination.getValue(now).toFixed(2) : "N/A"}° <br>
                    <strong>Density:</strong> ${entity.properties.density ? entity.properties.density.getValue(now).toExponential(2) : "N/A"} <br>
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

    // ensure all entities are not shown

    // initialise the model
    removeEntities();
    // document.getElementById('radio-leo').checked = true;
    handleOrbitToggle();

    // Define toggleOrbit to show/hide the orbit path.
    function toggleOrbit(entityId, colour) {
        const entity = dataSource && dataSource.entities && dataSource.entities.getById 
            ? dataSource.entities.getById(entityId)
            : null;
        if (!entity) return;
        if (entity.path) {
            removeEntityPath(entity);
        } else {
            showEntityPath(entity, colour);
        }
    }

    window.toggleOrbit = toggleOrbit;
    
    function showEntityPath(entity, colour) {
        let orbit_colour;
        
        // Default color (yellow) to use when RGB properties are missing
        const DEFAULT_COLOR = Cesium.Color.YELLOW;
        
        if (colour) {
            // Use the passed colour if provided (for override cases)
            orbit_colour = colour;
        } else {
            // Try to get the orbit colour from CZML properties, fallback to default if missing
            if (entity.properties && 
                entity.properties.orbit_colour_r && 
                entity.properties.orbit_colour_g && 
                entity.properties.orbit_colour_b) {
                
                const now = Cesium.JulianDate.now();
                
                try {
                    let r = entity.properties.orbit_colour_r.getValue(now);
                    let g = entity.properties.orbit_colour_g.getValue(now);
                    let b = entity.properties.orbit_colour_b.getValue(now);

                    // Validate RGB values
                    if (typeof r === 'number' && typeof g === 'number' && typeof b === 'number' &&
                        r >= 0 && r <= 255 && g >= 0 && g <= 255 && b >= 0 && b <= 255) {
                        // Create Cesium Color from RGB values (0-255 range)
                        orbit_colour = Cesium.Color.fromBytes(r, g, b, 255);
                    } else {
                        // Invalid RGB values, use default
                        orbit_colour = DEFAULT_COLOR;
                    }
                } catch (error) {
                    // Error getting RGB values, use default
                    console.warn(`Failed to get RGB values for entity ${entity.id}: ${error.message}, using default color`);
                    orbit_colour = DEFAULT_COLOR;
                }
            } else {
                // RGB properties missing, use default color
                orbit_colour = DEFAULT_COLOR;
            }
        }
        
        // Store the colour on the entity for later toggling
        entity.orbitColor = orbit_colour; // Fixed typo: was orbitcolour
        
        // Create or update the entity path with the correct colour
        if (entity.path) {
            entity.path.material = new Cesium.ColorMaterialProperty(orbit_colour); // Fixed typo: was colourMaterialProperty
            entity.path.width = 2;
            entity.path.show = true;
        } else {
            entity.path = new Cesium.PathGraphics({
                show: true,
                material: new Cesium.ColorMaterialProperty(orbit_colour), // Fixed typo: was colourMaterialProperty
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

    function removeEntities() {
        // Clear any manually added orbit paths
        viewer.entities.removeAll();
        
        // Also hide dataSource entities if needed
        dataSource.entities.values.forEach(entity => entity.show = false);
    }

    function handleOrbitToggle() {
        removeEntities();
        dataSource.entities.values.forEach(entity => {
            try {
                entity.show = true;
                showEntityPath(entity);
            } catch (error) {
                // Log error but continue processing other entities
                console.error(`Error showing orbit for entity ${entity.id}: ${error.message}`);
                // Still show the entity even if orbit path fails
                entity.show = true;
            }
        });
    }
});