//$(document).ready(function() {
//  $('#fileInput').change(function() {
//    $('#fileName').text($(this).prop('files')[0].name);
//  });
//
//  $('#submitBtn').click(function(e) {
//    e.preventDefault(); // Prevent default form submission
//
//    var formData = new FormData($('#fileForm')[0]);
//
//    $.ajax({
//      url: '/',
//      type: 'POST',
//      data: formData,
//      contentType: false,
//      processData: false,
//      xhr: function() {
//        var xhr = new window.XMLHttpRequest();
//        xhr.upload.addEventListener('progress', function(e) {
//          if (e.lengthComputable) {
//            var percent = (e.loaded / e.total) * 100;
//            $('#progressBar').attr('value', percent.toFixed(2)); // Update progress bar value
//          }
//        });
//        return xhr;
//      },
//      success: function(response) {
//        if (response.results) {
//          var results = response.results;
//          var resultHtml = '<div class="content"><ul>';
//          results.forEach(function(result) {
//            resultHtml += '<li>' + result + '</li>';
//          });
//          resultHtml += '</ul></div>';
//          $('#result').html(resultHtml); // Update only the result section
//          $('#myModal').addClass('is-active'); // Open modal
//          $('#modalResult').html(resultHtml); // Update modal content
//        } else {
//          $('#result').html('<div class="notification is-info">No results found</div>');
//        }
//      },
//      error: function() {
//        $('#result').html('<div class="notification is-danger">Error occurred while processing the request</div>');
//      },
//      complete: function() {
//        $('#progressBar').attr('value', 0); // Reset progress bar
//      }
//    });
//  });
//
//  $('.modal-close').click(function() {
//    $('#myModal').removeClass('is-active'); // Close modal
//  });
//});


$(document).ready(function() {
  $('#fileInput').change(function() {
    $('#fileName').text($(this).prop('files')[0].name);
  });

  $('#submitBtn').click(function(e) {
    e.preventDefault(); // Prevent default form submission

    var formData = new FormData($('#fileForm')[0]);

    $.ajax({
      url: '/',
      type: 'POST',
      data: formData,
      contentType: false,
      processData: false,
      xhr: function() {
        var xhr = new window.XMLHttpRequest();
        xhr.upload.addEventListener('progress', function(e) {
          if (e.lengthComputable) {
            var percent = (e.loaded / e.total) * 100;
            $('#progressBar').attr('value', percent.toFixed(2)); // Update progress bar value
          }
        });
        return xhr;
      },
      success: function(response) {
        if (response.results) {
          var results = response.results;
          var resultHtml = '<div class="content"><ul>';
          results.forEach(function(result) {
            resultHtml += '<li>' + result + '</li>';
          });
          resultHtml += '</ul></div>';
          $('#result').html(resultHtml); // Update only the result section
          $('#myModal').addClass('is-active'); // Open modal
          $('#modalResult').html(resultHtml); // Update modal content
        } else {
          $('#result').html('<div class="notification is-info">No results found</div>');
        }
      },
      error: function() {
        $('#result').html('<div class="notification is-danger">Error occurred while processing the request</div>');
      },
      complete: function() {
        $('#progressBar').attr('value', 0); // Reset progress bar
      }
    });
  });

  $('.modal-close').click(function() {
    $('#myModal').removeClass('is-active'); // Close modal
  });
});
