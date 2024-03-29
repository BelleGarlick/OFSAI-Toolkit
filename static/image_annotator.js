var selectedIndex = 0;
var zoom = 1;
var imageData = null;

let cornerAnnotationRadius = 8;

var relX = 0;
var relY = 0;

var selectedAnnotation = -1;
var hoveredAnnotation = -1;

var leftMouseDown = false;
var rightMouseDown = false

var imageOffsetX = 0
var imageOffsetY = 0

var topLeftCornerSelected = false;
var topRightCornerSelected = false;
var bottomLeftCornerSelected = false;
var bottomRightCornerSelected = false;

var initialBoxX = -1
var initialBoxY = -1

function loadImageFromJson() {
    if (imageData.image_found) {
        document.title = imageData.name
        let svg = $("#svg")

        center()

        svg.attr("viewBox", "0 0 " + imageData.width + " " + imageData.height)
        svg.css('background-image', "url('/img/" + imageData.name + "')");

        svg.mousemove(function(event) {
            let curX = event.offsetX / svg.width();
            let curY = event.offsetY / svg.height();

            let delX = (curX - relX)
            let delY = (curY - relY)

            relX = curX
            relY = curY

            if (rightMouseDown || (event.shiftKey && leftMouseDown)) {
                imageOffsetX += ($("#svg").width()) * delX
                imageOffsetY += ($("#svg").height()) * delY

                relX -= delX
                relY -= delY
            } else if (leftMouseDown) {
                if (selectedAnnotation != -1) {
                    if (topLeftCornerSelected) {
                        imageData.annotations[selectedAnnotation][1] += delX / 2
                        imageData.annotations[selectedAnnotation][2] += delY / 2
                        imageData.annotations[selectedAnnotation][3] -= delX
                        imageData.annotations[selectedAnnotation][4] -= delY
                    } else if (topRightCornerSelected) {
                        imageData.annotations[selectedAnnotation][1] += delX / 2
                        imageData.annotations[selectedAnnotation][2] += delY / 2
                        imageData.annotations[selectedAnnotation][3] += delX
                        imageData.annotations[selectedAnnotation][4] -= delY
                    } else if (bottomLeftCornerSelected) {
                        imageData.annotations[selectedAnnotation][1] += delX / 2
                        imageData.annotations[selectedAnnotation][2] += delY / 2
                        imageData.annotations[selectedAnnotation][3] -= delX
                        imageData.annotations[selectedAnnotation][4] += delY
                    } else if (bottomRightCornerSelected) {
                        imageData.annotations[selectedAnnotation][1] += delX / 2
                        imageData.annotations[selectedAnnotation][2] += delY / 2
                        imageData.annotations[selectedAnnotation][3] += delX
                        imageData.annotations[selectedAnnotation][4] += delY
                    } else {
                        imageData.annotations[selectedAnnotation][1] += delX
                        imageData.annotations[selectedAnnotation][2] += delY
                    }
                    imageData.annotations[selectedAnnotation][3] = Math.max(0.001, imageData.annotations[selectedAnnotation][3])
                    imageData.annotations[selectedAnnotation][4] = Math.max(0.001, imageData.annotations[selectedAnnotation][4])
                } else {
                    if (initialBoxX == -1 && initialBoxY == -1) {
                        initialBoxX = relX
                        initialBoxY = relY
                    } else if (Math.abs(initialBoxX - relX) * svg.width() > 3 && Math.abs(initialBoxY - relY) * svg.height()) {
                        let minX = (initialBoxX + relX) / 2
                        let minY = (initialBoxY + relY) / 2
                        let width = Math.abs(relX - initialBoxX)
                        let height = Math.abs(relY - initialBoxY)

                        imageData.annotations.push([
                            selectedIndex,
                            minX,
                            minY,
                            Math.max(width, 0.001),
                            Math.max(height, 0.001)
                        ])
                        selectedAnnotation = imageData.annotations.length - 1

                        topLeftCornerSelected = relX < initialBoxX && relY < initialBoxY
                        topRightCornerSelected = relX >= initialBoxX && relY < initialBoxY
                        bottomLeftCornerSelected = relX < initialBoxX && relY >= initialBoxY
                        bottomRightCornerSelected = relX >= initialBoxX && relY >= initialBoxY

                        initialBoxX = -1
                        initialBoxY = -1
                    }
                }
            } else {
                hoveredAnnotation = getAnnotationFromMousePos(relX, relY)
            }

            refreshSVG()
        });

        $("#svgContainer").bind('mousewheel', function(e) {
            if (e.originalEvent.deltaY < 0) {
                var scroll = Math.pow(1.001, Math.abs(e.originalEvent.deltaY))
                zoomImage(scroll)
            }
            if (e.originalEvent.deltaY > 0) {
                var scroll = Math.pow(0.999, Math.abs(e.originalEvent.deltaY))
                zoomImage(scroll)
            }
        });

        svg.mousedown(function(event) {
            if (event.originalEvent.button == 0) {onLeftMouseDown();}
            if (event.originalEvent.button == 2) {onRightMouseDown();}
            refreshSVG();
        });

        svg.mouseup(function(event) {
            if (event.originalEvent.button == 0) {onLeftMouseUp();}
            if (event.originalEvent.button == 2) {onRightMouseUp();}
            refreshSVG();
        });

        $("#svgContainer").mouseleave(function() {
            rightMouseDown = false
            leftMouseDown = false
        });

        refreshSVG();
    } else {
        document.getElementById('svgContainer').innerHTML = "<p style='padding: 30px;'>" + imageData.reason + "</p>"
    }
}

function zoomImage(scrollFactor) {
    zoom *= scrollFactor

    let svg = $("#svg")
    imageOffsetX -= (((relX - 0.5) * svg.width()) * scrollFactor) - ((relX - 0.5) * svg.width())
    imageOffsetY -= (((relY - 0.5) * svg.height()) * scrollFactor) - ((relY - 0.5) * svg.height())
    refreshSVG()
}

function onLeftMouseDown() {
    if (!leftMouseDown && !rightMouseDown) {
        leftMouseDown = true

        if (distanceToSelectedCornerTopLeft() != null && distanceToSelectedCornerTopLeft() < cornerAnnotationRadius * zoom) {
            topLeftCornerSelected = true
        } else if (distanceToSelectedCornerTopRight() != null && distanceToSelectedCornerTopRight() < cornerAnnotationRadius * zoom) {
            topRightCornerSelected = true
        } else if (distanceToSelectedCornerBottomLeft() != null && distanceToSelectedCornerBottomLeft() < cornerAnnotationRadius * zoom) {
            bottomLeftCornerSelected = true
        } else if (distanceToSelectedCornerBottomRight() != null && distanceToSelectedCornerBottomRight() < cornerAnnotationRadius * zoom) {
            bottomRightCornerSelected = true
        } else if (hoveredAnnotation != -1) {
            selectedAnnotation = hoveredAnnotation
            selectAnnotationIndex(imageData.annotations[selectedAnnotation][0])
        } else {
            selectedAnnotation = -1
        }
    }
}

function onLeftMouseUp() {
    leftMouseDown = false

    topLeftCornerSelected = false;
    topRightCornerSelected = false;
    bottomLeftCornerSelected = false;
    bottomRightCornerSelected = false;

    initialBoxX = -1
    initialBoxY = -1
}

function onRightMouseDown() {
    rightMouseDown = true
}

function onRightMouseUp() {
    rightMouseDown = false
}

function distanceToSelectedCornerTopLeft() {
    if (selectedAnnotation < 0) {return null}
    let top = imageData.height * (imageData.annotations[selectedAnnotation][2] - imageData.annotations[selectedAnnotation][4] / 2);
    let left = imageData.width * (imageData.annotations[selectedAnnotation][1] - imageData.annotations[selectedAnnotation][3] / 2);

    let absX = imageData.width * relX
    let absY = imageData.height * relY

    return Math.hypot(absX - left, absY - top)
}

function distanceToSelectedCornerTopRight() {
    if (selectedAnnotation < 0) {
        return null
    }
    let top = imageData.height * (imageData.annotations[selectedAnnotation][2] - imageData.annotations[selectedAnnotation][4] / 2);
    let right = imageData.width * (imageData.annotations[selectedAnnotation][1] + imageData.annotations[selectedAnnotation][3] / 2);

    let absX = imageData.width * relX
    let absY = imageData.height * relY

    return Math.hypot(absX - right, absY - top)
}

function distanceToSelectedCornerBottomLeft() {
    if (selectedAnnotation < 0) {
        return null
    }
    let bottom = imageData.height * (imageData.annotations[selectedAnnotation][2] + imageData.annotations[selectedAnnotation][4] / 2);
    let left = imageData.width * (imageData.annotations[selectedAnnotation][1] - imageData.annotations[selectedAnnotation][3] / 2);

    let absX = imageData.width * relX
    let absY = imageData.height * relY

    return Math.hypot(absX - left, absY - bottom)
}

function distanceToSelectedCornerBottomRight() {
    if (selectedAnnotation < 0) {
        return null
    }
    let bottom = imageData.height * (imageData.annotations[selectedAnnotation][2] + imageData.annotations[selectedAnnotation][4] / 2);
    let right = imageData.width * (imageData.annotations[selectedAnnotation][1] + imageData.annotations[selectedAnnotation][3] / 2);

    let absX = imageData.width * relX
    let absY = imageData.height * relY

    return Math.hypot(absX - right, absY - bottom)
}

function refreshSVG() {
    let svg = $("#svg")
    let svgCode = "";

    // Draw Boxes
    let selectedAnnotationCode = ""
    for (var a = 0; a < imageData.annotations.length; a++) {
        let classNum = imageData.annotations[a][0]
        let x = (imageData.annotations[a][1] - imageData.annotations[a][3] / 2) * imageData.width
        let y = (imageData.annotations[a][2] - imageData.annotations[a][4] / 2) * imageData.height
        let w = (imageData.annotations[a][3]) * imageData.width
        let h = (imageData.annotations[a][4]) * imageData.height

        let alpha = 0.2;
        if (a == hoveredAnnotation) {
            alpha = 0.5
        }

        let dashStrokeSpacing = 20
        if (a == selectedAnnotation) {
            alpha = 0.5
            dashStrokeSpacing = 0

            let l = (imageData.annotations[a][1] - imageData.annotations[a][3] / 2) * imageData.width
            let r = (imageData.annotations[a][1] + imageData.annotations[a][3] / 2) * imageData.width
            let t = (imageData.annotations[a][2] - imageData.annotations[a][4] / 2) * imageData.height
            let b = (imageData.annotations[a][2] + imageData.annotations[a][4] / 2) * imageData.height
            selectedAnnotationCode += '<circle cx="'+l+'" cy="'+t+'" r="'+ (cornerAnnotationRadius / zoom) +'" style="fill:'+classColours[classNum]+';"></circle>';
            selectedAnnotationCode += '<circle cx="'+r+'" cy="'+t+'" r="'+ (cornerAnnotationRadius / zoom) +'" style="fill:'+classColours[classNum]+';"></circle>';
            selectedAnnotationCode += '<circle cx="'+l+'" cy="'+b+'" r="'+ (cornerAnnotationRadius / zoom) +'" style="fill:'+classColours[classNum]+';"></circle>';
            selectedAnnotationCode += '<circle cx="'+r+'" cy="'+b+'" r="'+ (cornerAnnotationRadius / zoom) +'" style="fill:'+classColours[classNum]+';"></circle>';
            selectedAnnotationCode += '<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" style="fill:'+classColours[classNum]+';" stroke="'+classColours[classNum]+'" stroke-width="'+3/zoom+'" stroke-dasharray="20, '+dashStrokeSpacing+'" fill-opacity="' +alpha+ '"></rect>';
        } else {
            svgCode += '<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" style="fill:'+classColours[classNum]+';" stroke="'+classColours[classNum]+'" stroke-width="'+3/zoom+'" stroke-dasharray="20, '+dashStrokeSpacing+'" fill-opacity="' +alpha+ '"></rect>';
        }
    }
    svgCode += selectedAnnotationCode

    // Calculate cursor lines
    cursorX = relX * imageData.width
    cursorY = relY * imageData.height
    svgCode += '<line x1="'+cursorX+'" y1="0" x2="'+cursorX+'" y2="'+imageData.height+'" stroke-dasharray="10,10" style="stroke: {{cursor_line_colour}}; stroke-width:'+{{cursor_line_width}}/zoom+'"/>'
    svgCode += '<line x1="0" y1="'+cursorY+'" x2="'+imageData.width+'" y2="'+cursorY+'" stroke-dasharray="10,10" style="stroke: {{cursor_line_colour}}; stroke-width:'+{{cursor_line_width}}/zoom+'"/>'

    // Zoom / position
    svg.height(imageData.height * zoom)
    svg.width(imageData.width * zoom)

    let parentWidth = svg.parent().width()
    let parentHeight = svg.parent().height()
    $("#svgContainer").css({"height": (window.innerHeight-60)+"px"})
    svg.css({"top": (imageOffsetY + (parentHeight-svg.height())/2) + "px", "left": (imageOffsetX + (parentWidth-svg.width())/2) + "px"})

    svg.html(svgCode)
}

function center() {
    let svg = $("#svg")

    let imWidth = imageData.width
    let imHeight = imageData.height

    let parentWidth = svg.parent().width()
    let parentHeight = svg.parent().height()

    zoom = Math.min(parentHeight / imHeight, parentWidth / imWidth)
    imageOffsetX = 0
    imageOffsetY = 0
}

function saveImage() {
    $.post({{save_route}}, {"image_data": JSON.stringify(imageData)}, function(data) {
        if (data.success) {
            refresh()
        } else {
            alert(data.reason)
        }
    });
}

function selectAnnotationIndex(index) {
    selectedIndex = Math.max(0, Math.min(index, numClasses - 1))

    $("#annotationContainer").children().each(function(annotationIndex) {
        let annoColour = $(this).attr("color")
        $(this).css({"color": annoColour, "background": "white"});
        if (annotationIndex == selectedIndex) {
            $(this).css({"color": "white", "background": annoColour});
        }
    });

    if (selectedAnnotation != -1) {
        imageData.annotations[selectedAnnotation][0] = index;
        refreshSVG()
    }
}

window.onkeydown = function(e) {
    if (e.keyCode >= 49 && e.keyCode <= 57) {selectAnnotationIndex(e.keyCode - 49)}
    if (e.keyCode == 13) {saveImage();}
    if (e.keyCode == 78 || e.keyCode == 32) {refresh()}
    if (e.keyCode == 27) {selectedAnnotation = -1; refreshSVG()}
    if (e.keyCode == 8 || e.keyCode == 46) {deleteSelectedAnnotation(); refreshSVG()}
    if (e.keyCode == 82) {center(); refreshSVG()}
    if (e.keyCode == 83) {saveImage()}
}

function getAnnotationFromMousePos(relX, relY) {
    for (let an = 0; an < imageData.annotations.length; an++) {
        let annotation = imageData.annotations[an];
        if (annotation[1] - annotation[3]/2 <= relX && annotation[1] + annotation[3]/2 >= relX && annotation[2] - annotation[4]/2 <= relY && annotation[2] + annotation[4]/2 >= relY) {
            return an
        }
    }
    return -1
}

function deleteSelectedAnnotation() {
    if (selectedAnnotation != -1) {
        imageData.annotations.splice(selectedAnnotation, 1)
        selectedAnnotation = -1
        refreshSVG()
    }
}

window.onload = function() {
    $.get("{{new_route}}", function(d, status) {
        imageData = JSON.parse(d)
        loadImageFromJson()
    })
    selectAnnotationIndex(0)
}

window.onresize = function() {
    refreshSVG()
}

function refresh() {location.href = location.href;}
document.addEventListener('contextmenu', event => event.preventDefault());