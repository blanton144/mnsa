$.ajaxSetup({
				async: false
});

const urlSearchParams = new URLSearchParams(window.location.search);
const params = Object.fromEntries(urlSearchParams.entries());

var plate = "";
var ifu = "";
var plateifu = {};

var ifus = ['1901', '1902', '3701', '3702', '3703', '3704',
						'6101', '6102', '6103', '6104', '9101', '9102',
						'12701', '12702', '12703', '12704', '12705']

function setPlateIFU(in_plate, in_ifu) {
		plate = in_plate;
		ifu = in_ifu;
		plateifu = plateifus[plate + '-' + ifu];
		setPlateIFUTableInfo();
		setPlateIFUImages();
		setPlateIFUFormInfo();
}

function setPlateIFUFormInfo() {
    document.getElementById("plate_input").placeholder = plate;
    document.getElementById("ifu_input").placeholder = ifu;
}

function setPlateIFUTableInfo() {
    document.getElementById("plate").innerHTML = plate;
    document.getElementById("ifu").innerHTML = ifu;
		document.getElementById("ra").innerHTML = plateifu['ifura'];
		document.getElementById("dec").innerHTML = plateifu['ifudec'];
}

function setPlateIFUImages() {
		crr_image = `<a href="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-irg.png"><img class="image" src="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-irg.png"/></a>`;
    document.getElementById("crr_image").innerHTML = crr_image;
		orig_image = `<a href="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-dr17-irg.png"><img class="image" src="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-dr17-irg.png"/></a>`;
    document.getElementById("orig_image").innerHTML = orig_image;
		mnsa_Ha_O3_Grey = `<a href="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-Ha-O3-Grey.png"><img class="image" src="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-Ha-O3-Grey.png"/></a>`;
    document.getElementById("mnsa_Ha_O3_Grey").innerHTML = mnsa_Ha_O3_Grey;
		dr17_Ha_O3_Grey = `<a href="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-dr17-Ha-O3-Grey.png"><img class="image" src="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-dr17-Ha-O3-Grey.png"/></a>`;
    document.getElementById("dr17_Ha_O3_Grey").innerHTML = dr17_Ha_O3_Grey;
		desi_image = `<a href="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-zrg-native.png"><img class="image" src="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-zrg-native.png"/></a>`;
    document.getElementById("desi_image").innerHTML = desi_image;
		desi_image_resampled = `<a href="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-dlis.png"><img class="image" src="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-dlis.png"/></a>`;
    document.getElementById("desi_image_resampled").innerHTML = desi_image_resampled;
		wise_image_resampled = `<a href="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-wise.png"><img class="image" src="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-wise.png"/></a>`;
    document.getElementById("wise_image_resampled").innerHTML = wise_image_resampled;
		galex_image_resampled = `<a href="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-galex.png"><img class="image" src="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-galex.png"/></a>`;
    document.getElementById("galex_image_resampled").innerHTML = galex_image_resampled;
		sps_spec = `<a href="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-sps-AP06.png"><img class="sps_spec_img" src="pngs/${plate}/${plate}-${ifu}/manga-${plate}-${ifu}-sps-AP06.png"/></a>`;
    document.getElementById("sps_spec").innerHTML = sps_spec;
}

function plateifu_autocomplete(inp, arr) {
  /*the autocomplete function takes two arguments,
  the text field element and an array of possible autocompleted values:*/
  var currentFocus;
  /*execute a function when someone writes in the text field:*/
  inp.addEventListener("input", function(e) {
      var a, b, i, val = this.value;
      /*close any already open lists of autocompleted values*/
      closeAllLists();
      if (!val) { return false;}
      currentFocus = -1;
      /*create a DIV element that will contain the items (values):*/
      a = document.createElement("DIV");
      a.setAttribute("id", this.id + "autocomplete-list");
      a.setAttribute("class", "autocomplete-items");
      /*append the DIV element as a child of the autocomplete container:*/
      this.parentNode.appendChild(a);
      /*for each item in the array...*/
      for (i = 0; i < arr.length; i++) {
        /*check if the item starts with the same letters as the text field value:*/
        if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
          /*create a DIV element for each matching element:*/
          b = document.createElement("DIV");
          /*make the matching letters bold:*/
          b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
          b.innerHTML += arr[i].substr(val.length);
          /*insert a input field that will hold the current array item's value:*/
          b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
          /*execute a function when someone clicks on the item value (DIV element):*/
              b.addEventListener("click", function(e) {
              /*insert the value for the autocomplete text field:*/
              inp.value = this.getElementsByTagName("input")[0].value;
              /*close the list of autocompleted values,
              (or any other open lists of autocompleted values:*/
              closeAllLists();
          });
          a.appendChild(b);
        }
      }
  });
  /*execute a function presses a key on the keyboard:*/
  inp.addEventListener("keydown", function(e) {
      var x = document.getElementById(this.id + "autocomplete-list");
      if (x) x = x.getElementsByTagName("div");
      if (e.keyCode == 40) {
        /*If the arrow DOWN key is pressed,
        increase the currentFocus variable:*/
        currentFocus++;
        /*and and make the current item more visible:*/
        addActive(x);
      } else if (e.keyCode == 38) { //up
        /*If the arrow UP key is pressed,
        decrease the currentFocus variable:*/
        currentFocus--;
        /*and and make the current item more visible:*/
        addActive(x);
      } else if (e.keyCode == 13) {
        /*If the ENTER key is pressed, prevent the form from being submitted,*/
        e.preventDefault();
        if (currentFocus > -1) {
          /*and simulate a click on the "active" item:*/
          if (x) x[currentFocus].click();
        }
      }
  });
  function addActive(x) {
    /*a function to classify an item as "active":*/
    if (!x) return false;
    /*start by removing the "active" class on all items:*/
    removeActive(x);
    if (currentFocus >= x.length) currentFocus = 0;
    if (currentFocus < 0) currentFocus = (x.length - 1);
    /*add class "autocomplete-active":*/
    x[currentFocus].classList.add("autocomplete-active");
  }
  function removeActive(x) {
    /*a function to remove the "active" class from all autocomplete items:*/
    for (var i = 0; i < x.length; i++) {
      x[i].classList.remove("autocomplete-active");
    }
  }
  function closeAllLists(elmnt) {
    /*close all autocomplete lists in the document,
    except the one passed as an argument:*/
    var x = document.getElementsByClassName("autocomplete-items");
    for (var i = 0; i < x.length; i++) {
      if (elmnt != x[i] && elmnt != inp) {
      x[i].parentNode.removeChild(x[i]);
    }
  }
}
/*execute a function when someone clicks in the document:*/
document.addEventListener("click", function (e) {
    closeAllLists(e.target);
});
}

plateifu_autocomplete(document.getElementById("plate_input"), plates);
plateifu_autocomplete(document.getElementById("ifu_input"), ifus);

setPlateIFU(params.plate, params.ifu);
